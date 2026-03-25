"""Global hotkey management for the application."""

from __future__ import annotations

import logging
import platform
import threading
from typing import Callable

import keyboard


logger = logging.getLogger(__name__)

# 检测当前平台
CURRENT_PLATFORM = platform.system().lower()  # "linux", "windows", "darwin"


def _convert_keyboard_to_pynput(combo: str) -> str:
    """将 keyboard 库的热键格式转换为 pynput GlobalHotKeys 格式。

    keyboard 格式: "f2", "ctrl+shift+h", "alt+f4"
    pynput 格式: "<f2>", "<ctrl>+<shift>+h", "<alt>+<f4>"

    Args:
        combo: keyboard 格式的热键字符串

    Returns:
        pynput 格式的热键字符串
    """
    # 定义需要用 <> 包裹的键
    special_keys = {
        'ctrl', 'shift', 'alt', 'cmd', 'win',
        'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9',
        'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17',
        'f18', 'f19', 'f20',
        'up', 'down', 'left', 'right',
        'enter', 'space', 'tab', 'esc', 'escape', 'backspace',
        'delete', 'del', 'home', 'end', 'page_up', 'page_down',
        'pgup', 'pgdn', 'insert', 'caps_lock', 'num_lock',
        'scroll_lock', 'print_screen', 'pause', 'break',
    }

    parts = combo.lower().replace(' ', '').split('+')
    converted = []

    for part in parts:
        part = part.strip()
        if part in special_keys:
            converted.append(f'<{part}>')
        else:
            converted.append(part)

    return '+'.join(converted)


class HotkeyManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._registrations = {}
        # 根据平台选择实现：Linux 使用 pynput，其他平台使用 keyboard
        self._use_pynput = CURRENT_PLATFORM == "linux"

        if self._use_pynput:
            self._pynput_listener = None
            self._pynput_callbacks = {}
            self._pynput_wait_event = threading.Event()
            logger.info("使用 pynput 库进行热键管理 (平台: %s)", CURRENT_PLATFORM)
        else:
            logger.info("使用 keyboard 库进行热键管理 (平台: %s)", CURRENT_PLATFORM)

    def _init_pynput(self) -> None:
        """初始化 pynput 库（延迟导入以避免不必要的依赖）。"""
        if self._pynput_listener is not None:
            return

        try:
            from pynput.keyboard import GlobalHotKeys
            self._GlobalHotKeys = GlobalHotKeys
        except ImportError as exc:
            logger.error("pynput 库未安装，请运行: pip install pynput")
            raise RuntimeError("Linux 平台需要 pynput 库支持") from exc

    def _pynput_register(self, combo: str, callback: Callable[[], None]) -> None:
        """使用 pynput 注册热键。"""
        self._init_pynput()

        # 转换热键格式
        pynput_combo = _convert_keyboard_to_pynput(combo)

        # 存储回调
        self._pynput_callbacks[pynput_combo] = callback

        # 重建监听器（pynput 不支持动态添加，需要重建）
        if self._pynput_listener is not None:
            self._pynput_listener.stop()

        hotkeys_map = {combo: cb for combo, cb in self._pynput_callbacks.items()}
        self._pynput_listener = self._GlobalHotKeys(hotkeys_map)
        self._pynput_listener.start()

        logger.info("已注册热键 %s (pynput 格式: %s)", combo, pynput_combo)

    def _pynput_unregister_all(self) -> None:
        """移除所有 pynput 热键。"""
        if self._pynput_listener is not None:
            self._pynput_listener.stop()
            self._pynput_listener = None

        self._pynput_callbacks.clear()
        logger.info("已移除所有 pynput 热键")

    def register(self, combo: str, callback: Callable[[], None]) -> None:
        with self._lock:
            if self._use_pynput:
                # Linux 平台使用 pynput
                if combo in self._registrations:
                    logger.warning("热键 %s 已注册，覆盖旧的回调", combo)

                try:
                    self._pynput_register(combo, callback)
                except Exception as exc:  # noqa: BLE001
                    logger.error("注册热键 %s 失败: %s", combo, exc)
                    raise

                self._registrations[combo] = callback
            else:
                # Windows/macOS 平台使用 keyboard
                if combo in self._registrations:
                    logger.warning("热键 %s 已注册，覆盖旧的回调", combo)
                    keyboard.remove_hotkey(self._registrations[combo])

                try:
                    hotkey_id = keyboard.add_hotkey(combo, callback)
                except Exception as exc:  # noqa: BLE001
                    logger.error("注册热键 %s 失败: %s", combo, exc)
                    raise

                self._registrations[combo] = hotkey_id
                logger.info("已注册热键 %s", combo)

    def unregister_all(self) -> None:
        with self._lock:
            if self._use_pynput:
                # Linux 平台使用 pynput
                self._pynput_unregister_all()
            else:
                # Windows/macOS 平台使用 keyboard
                for combo, hotkey_id in list(self._registrations.items()):
                    keyboard.remove_hotkey(hotkey_id)
                    logger.info("已移除热键 %s", combo)

            self._registrations.clear()

    def cleanup(self) -> None:
        self.unregister_all()

        if self._use_pynput:
            # pynput 清理
            self._pynput_unregister_all()
        else:
            # keyboard 清理：彻底停止 keyboard 库的所有钩子和监听线程
            try:
                keyboard.unhook_all()
                logger.info("已停止 keyboard 监听线程")
            except Exception as exc:
                logger.warning("停止 keyboard 监听线程失败: %s", exc)

    def wait(self) -> None:
        """保持程序运行，等待热键触发。"""
        if self._use_pynput:
            # pynput: 使用 threading.Event 阻塞
            logger.debug("使用 pynput 等待模式")
            self._pynput_wait_event.wait()
        else:
            # keyboard: 使用 keyboard.wait()
            logger.debug("使用 keyboard 等待模式")
            keyboard.wait()


