#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FunASR 模型下载脚本
并行下载所有模型文件
"""
import logging
import sys
import json
import threading
import os
from pathlib import Path
from app.funasr_config import MODEL_REVISION, get_models_for_download
from app.logging_config import setup_logging

logger = logging.getLogger(__name__)


def get_custom_model_dir(model_type):
    """
    获取自定义模型目录
    优先级：FUNASR_MODEL_DIR 环境变量 > 用户主目录/.vocotype/models
    """
    # 1. 检查环境变量（最高优先级）
    custom_dir = os.environ.get("FUNASR_MODEL_DIR")
    if custom_dir:
        logger.info(f"检测到模型目录环境变量：{custom_dir}")
        model_dir = Path(custom_dir) / model_type
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    # 2. 使用用户主目录下的固定位置（打包后也稳定）
    #    Windows: C:\Users\<用户名>\.vocotype\models
    home = Path.home()
    model_dir = home / ".vocotype" / "models" / model_type
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def download_model(model_config, progress_callback=None):
    """下载单个模型（使用 modelscope.snapshot_download，支持自定义目录）"""
    model_name = model_config["name"]
    model_type = model_config["type"]

    try:
        from modelscope.hub.snapshot_download import snapshot_download

        if progress_callback:
            progress_callback(model_type, "downloading", 0)

        # 获取自定义模型目录
        model_dir = get_custom_model_dir(model_type)

        # 下载到自定义目录
        snapshot_download(
            model_name,
            revision=MODEL_REVISION,
            local_dir=str(model_dir)
        )

        if progress_callback:
            progress_callback(model_type, "completed", 100)

        return {"success": True, "model": model_type, "path": str(model_dir)}

    except Exception as e:
        if progress_callback:
            progress_callback(model_type, "error", 0, str(e))
        return {"success": False, "model": model_type, "error": str(e)}

def main():
    """主函数：并行下载所有模型"""
    # 配置日志系统（使用统一配置）
    import os
    project_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(project_root, "logs")
    setup_logging("INFO", log_dir)

    # 从统一配置获取模型列表
    models = get_models_for_download()

    # 进度跟踪
    progress = {"asr": 0, "vad": 0, "punc": 0}
    results = {}
    completed_count = 0
    total_count = len(models)
    count_lock = threading.Lock()  # 添加锁保护计数器
    results_lock = threading.Lock()

    def progress_callback(model_type, stage, percent, error=None):
        nonlocal completed_count

        # 使用锁保护共享变量的修改
        with count_lock:
            if stage == "downloading":
                progress[model_type] = percent
            elif stage == "completed":
                progress[model_type] = 100
                completed_count += 1
            elif stage == "error":
                progress[model_type] = 0
                completed_count += 1

            # 计算总体进度
            overall_progress = sum(progress.values()) / total_count
            current_completed = completed_count

        # 输出进度信息（在锁外执行 I/O 操作）
        status = {
            "stage": stage,
            "model": model_type,
            "progress": percent,
            "overall_progress": round(overall_progress, 1),
            "completed": current_completed,
            "total": total_count
        }

        if error:
            status["error"] = error

        print(json.dumps(status, ensure_ascii=False))
        sys.stdout.flush()

    # 启动并行下载线程
    threads = []
    for model_config in models:
        def worker(config=model_config):
            result = download_model(config, progress_callback)
            with results_lock:
                results[config["type"]] = result

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        threads.append(thread)

    # 等待所有线程完成
    for thread in threads:
        thread.join()

    # 检查结果
    failed_models = [model_type for model_type, result in results.items() if not result["success"]]

    if failed_models:
        final_result = {
            "success": False,
            "error": f"以下模型下载失败：{', '.join(failed_models)}",
            "failed_models": failed_models,
            "results": results
        }
    else:
        final_result = {
            "success": True,
            "message": "所有模型下载完成",
            "results": results
        }

    print(json.dumps(final_result, ensure_ascii=False))
    sys.stdout.flush()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e)
        }
        print(json.dumps(error_result, ensure_ascii=False))
        sys.exit(1)


def get_model_cache_path(model_name, revision):
    """
    获取模型路径（支持自定义目录）
    1. 先检查自定义目录是否存在模型文件
    2. 如果存在，直接返回路径
    3. 如果不存在，调用 snapshot_download 进行下载到自定义目录
    """
    # 从完整模型名提取简短名称 (iic/xxx -> xxx)
    short_name = model_name.split('/')[-1] if '/' in model_name else model_name

    # 获取自定义模型目录（按模型类型分类）
    model_type_map = {
        "asr": "asr",
        "vad": "vad",
        "punc": "punc"
    }
    model_type = "asr"  # 默认
    for key, value in model_type_map.items():
        if key in short_name or (key == "asr" and "paraformer" in short_name):
            model_type = value
            break

    # 优先使用自定义目录
    model_dir = get_custom_model_dir(model_type)

    # 检查模型文件是否存在
    quant_file = model_dir / "model_quant.onnx"
    base_file = model_dir / "model.onnx"

    # 只要有一个模型文件存在，就认为缓存有效
    if quant_file.exists() or base_file.exists():
        logger.info(f"使用本地模型：{model_dir}")
        return str(model_dir)

    # 本地不存在，需要下载
    logger.info(f"本地模型不存在，开始下载模型：{model_name}")
    from modelscope.hub.snapshot_download import snapshot_download

    # 直接下载到自定义目录（不再使用离线模式回退）
    try:
        snapshot_download(
            model_name,
            revision=revision,
            local_dir=str(model_dir)
        )
        logger.info(f"模型下载完成到自定义目录：{model_dir}")
        return str(model_dir)
    except Exception as e:
        logger.error(f"模型下载失败：{e}")
        # 下载失败时，尝试使用默认缓存路径作为后备
        try:
            fallback_dir = snapshot_download(
                model_name,
                revision=revision,
            )
            logger.info(f"使用默认缓存路径作为后备：{fallback_dir}")
            return fallback_dir
        except Exception as fallback_error:
            logger.error(f"后备下载也失败：{fallback_error}")
            raise
