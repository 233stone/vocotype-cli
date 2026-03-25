"""Command-line entry for the speak-keyboard prototype."""

from __future__ import annotations

import argparse
import logging
import threading
import time

import keyboard

from app import HotkeyManager, TranscriptionResult, TranscriptionWorker, load_config, type_text
from app.plugins.dataset_recorder import wrap_result_handler
from app.logging_config import setup_logging


logger = logging.getLogger(__name__)


_TOGGLE_DEBOUNCE_SECONDS = 0.2
_toggle_lock = threading.Lock()
_last_toggle_time = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Speak Keyboard prototype")
    parser.add_argument("--config", default="config.json", help="Path to config JSON")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single transcription cycle for debugging",
    )
    parser.add_argument("--save-dataset", action="store_true", help="Persist audio/text pairs")
    parser.add_argument("--dataset-dir", default="dataset", help="Dataset output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    
    # 配置日志系统（统一配置）
    from app.config import ensure_logging_dir
    log_dir_abs = ensure_logging_dir(config)
    setup_logging(
        level=config["logging"].get("level", "INFO"),
        log_dir=log_dir_abs
    )

    output_cfg = config.get("output", {})
    output_method = output_cfg.get("method", "auto")
    append_newline = output_cfg.get("append_newline", False)

    # 先创建worker（没有回调）
    worker = TranscriptionWorker(
        config_path=args.config,
        on_result=None,  # 稍后设置
    )
    
    # 创建result handler（需要worker引用）
    worker.on_result = _make_result_handler(output_method, append_newline, worker)
    if args.save_dataset:
        worker.on_result = wrap_result_handler(worker.on_result, worker, args.dataset_dir)
    
    hotkeys = HotkeyManager()

    toggle_combo = config["hotkeys"].get("toggle", "f2")
    hotkeys.register(toggle_combo, lambda: _toggle(worker))

    try:
        logger.info("Speak Keyboard 启动完成，按 %s 开始/停止录音，按 Ctrl+C 退出", toggle_combo)
        if args.once:
            _toggle(worker)
            input("按 Enter 停止并退出...")
            _toggle(worker)
        else:
            hotkeys.wait()
    except KeyboardInterrupt:
        logger.info("用户中断，正在退出...")
    finally:
        # 清理所有资源
        try:
            worker.stop()
        except Exception as exc:
            logger.debug("停止 worker 时出错: %s", exc)
        
        try:
            worker.cleanup()
        except Exception as exc:
            logger.debug("清理 worker 时出错: %s", exc)
        
        try:
            # 显式清理 FunASR 模型资源
            worker.fun_server.cleanup()
        except Exception as exc:
            logger.debug("清理 FunASR 服务器时出错: %s", exc)
        
        try:
            hotkeys.cleanup()
        except Exception as exc:
            logger.debug("清理热键时出错: %s", exc)
        
        logger.info("所有资源已清理，正常退出")
        import sys
        sys.exit(0)


def _make_result_handler(output_method: str, append_newline: bool, worker: TranscriptionWorker):
    def _handle_result(result: TranscriptionResult) -> None:
        if result.error:
            logger.error("转写失败: %s", result.error)
            return

        # 获取转录统计信息
        stats = worker.transcription_stats
        
        logger.info(
            "转写成功: %s (推理 %.2fs) [已完成 %d/%d，队列剩余 %d]",
            result.text,
            result.inference_latency,
            stats["completed"],
            stats["submitted"],
            stats["pending"],
        )
        type_text(
            result.text,
            append_newline=append_newline,
            method=output_method,
        )

    return _handle_result


def _toggle(worker: TranscriptionWorker) -> None:
    global _last_toggle_time
    now = time.monotonic()
    with _toggle_lock:
        if now - _last_toggle_time < _TOGGLE_DEBOUNCE_SECONDS:
            logger.debug("忽略快速重复的录音切换请求 (%.3fs)", now - _last_toggle_time)
            return
        _last_toggle_time = now

    if worker.is_running:
        # 停止录音，提交转录任务
        worker.stop()
        stats = worker.transcription_stats
        if stats["pending"] > 0:
            logger.info(
                "录音已停止并提交转录，队列中还有 %d 个任务等待处理",
                stats["pending"]
            )
    else:
        # 开始录音
        stats = worker.transcription_stats
        if stats["pending"] > 0:
            logger.info(
                "开始录音（后台还有 %d 个转录任务正在处理）",
                stats["pending"]
            )
        worker.start()


if __name__ == "__main__":
    main()

