"""Text injection utilities for cross-platform support."""

from __future__ import annotations

import logging
import platform
import time

logger = logging.getLogger(__name__)

# 当前操作系统
CURRENT_PLATFORM = platform.system().lower()


def type_text(text: str, append_newline: bool = False, method: str = "auto") -> None:
    """跨平台文本注入函数。

    Args:
        text: 要输入的文本
        append_newline: 是否在文本末尾添加换行
        method: 输入方法 ("auto", "type", "clipboard", "legacy")
    """
    if not text:
        return

    payload = text + ("\n" if append_newline else "")
    logger.debug("📝 准备输出文本: %s", payload[:50] + "..." if len(payload) > 50 else payload)

    method = (method or "auto").lower()
    logger.debug("🔧 输出方法: %s", method)

    # Linux 平台：强制使用剪贴板模式
    if CURRENT_PLATFORM == "linux":
        if method != "clipboard" and method != "auto":
            logger.warning("⚠️ Linux 平台仅支持剪贴板输出，method='%s' 配置已忽略", method)
        order = ["clipboard"]
    elif method == "type":
        order = ["type", "clipboard"]
    elif method == "clipboard":
        order = ["clipboard", "type"]
    elif method == "legacy":
        order = ["legacy"] if CURRENT_PLATFORM == "windows" else ["type", "clipboard"]
    else:  # auto
        if CURRENT_PLATFORM == "windows":
            order = ["type", "clipboard", "legacy"]
        else:
            order = ["type", "clipboard"]

    logger.debug("🔄 尝试顺序: %s", order)

    for mode in order:
        try:
            logger.debug("  → 尝试模式: %s", mode)
            if mode == "type" and CURRENT_PLATFORM == "windows" and _type_with_legacy(payload):
                logger.info("✅ 使用 Windows legacy 输入成功")
                return
            if mode == "clipboard" and _try_clipboard_injection(payload):
                logger.debug("✅ 使用剪贴板输出成功")
                return
            if mode == "legacy" and CURRENT_PLATFORM == "windows" and _type_with_legacy(payload):
                logger.info("✅ 使用 legacy 方式成功")
                return
        except Exception as exc:
            logger.debug("模式 %s 失败: %s", mode, exc)
            continue

    logger.error("❌ 所有文本注入方式均失败: %s", payload[:50])

def _try_clipboard_injection(text: str) -> bool:
    """使用剪贴板进行文本注入。"""
    try:
        import pyperclip
    except ImportError:
        logger.debug("pyperclip 未安装")
        return False

    try:
        # 复制文本到剪贴板
        logger.debug("复制文本到剪贴板: %s", text[:50] + "..." if len(text) > 50 else text)
        pyperclip.copy(text)

        # 给剪贴板操作一点时间，确保复制完成
        time.sleep(0.15)

        # 验证剪贴板内容
        current_clip = pyperclip.paste()
        if current_clip != text:
            logger.warning("剪贴板验证失败: 期望 '%s', 实际 '%s'", text[:20], current_clip[:20])
            # 继续尝试，可能只是显示问题

        # 模拟粘贴操作
        logger.debug("模拟粘贴操作...")
        success = _simulate_ctrl_v()

        if success:
            logger.debug("✓ 剪贴板粘贴成功: %s", text[:30] + "..." if len(text) > 30 else text)
        else:
            logger.warning("剪贴板粘贴失败")

        # 不再恢复旧剪贴板内容，保持当前文本在剪贴板中
        # 这样用户可以手动粘贴，也不会影响下次使用

        return success
    except Exception as exc:
        logger.debug("剪贴板注入失败: %s", exc)
        return False


def _simulate_ctrl_v() -> bool:
    """模拟粘贴操作。

    终端使用 Ctrl+Shift+V，其他应用使用 Ctrl+V。

    注意：此函数使用 pynput 进行按键模拟，仅用于剪贴板粘贴功能，
    与热键监听功能共用 pynput 依赖。
    """
    try:
        from pynput.keyboard import Controller, Key

        keyboard = Controller()

        # 先尝试 Ctrl+Shift+V（终端通用）
        # 如果失败，回退到 Ctrl+V（GUI 应用）
        try:
            # 尝试 Ctrl+Shift+V
            with keyboard.pressed(Key.ctrl, Key.shift):
                keyboard.press('v')
                keyboard.release('v')
            time.sleep(0.05)
            logger.debug("使用 Ctrl+Shift+V 模拟粘贴（终端）")
            return True
        except Exception as e:
            # 回退到 Ctrl+V
            with keyboard.pressed(Key.ctrl):
                keyboard.press('v')
                keyboard.release('v')
            time.sleep(0.05)
            logger.debug("使用 Ctrl+V 模拟粘贴（GUI 应用）")
            return True

    except Exception as exc:
        logger.debug("模拟粘贴失败: %s", exc)
        return False


# ==================== Windows 特定实现 ====================

def _type_with_legacy(text: str) -> bool:
    """Windows 传统 ctypes 实现（保留作为备选）。"""
    if CURRENT_PLATFORM != "windows":
        return False

    try:
        import ctypes
        import ctypes.wintypes as wintypes

        SendInput = ctypes.windll.user32.SendInput
        GetMessageExtraInfo = ctypes.windll.user32.GetMessageExtraInfo

        INPUT_KEYBOARD = 1
        KEYEVENTF_KEYUP = 0x0002
        KEYEVENTF_UNICODE = 0x0004

        if hasattr(wintypes, "ULONG_PTR"):
            ULONG_PTR = wintypes.ULONG_PTR
        else:
            if ctypes.sizeof(ctypes.c_void_p) == ctypes.sizeof(ctypes.c_uint64):
                ULONG_PTR = ctypes.c_uint64
            else:
                ULONG_PTR = ctypes.c_uint32

        class KeyboardInput(ctypes.Structure):
            _fields_ = [
                ("wVk", wintypes.WORD),
                ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ULONG_PTR),
            ]

        class InputUnion(ctypes.Union):
            _fields_ = [("ki", KeyboardInput)]

        class INPUT(ctypes.Structure):
            _fields_ = [("type", wintypes.DWORD), ("union", InputUnion)]

        # 使用 Unicode 输入
        for char in text:
            code_point = ord(char)
            input_array_type = INPUT * 2
            inputs = input_array_type(
                INPUT(
                    type=INPUT_KEYBOARD,
                    union=InputUnion(
                        ki=KeyboardInput(
                            wVk=0,
                            wScan=code_point,
                            dwFlags=KEYEVENTF_UNICODE,
                            time=0,
                            dwExtraInfo=GetMessageExtraInfo(),
                        )
                    ),
                ),
                INPUT(
                    type=INPUT_KEYBOARD,
                    union=InputUnion(
                        ki=KeyboardInput(
                            wVk=0,
                            wScan=code_point,
                            dwFlags=KEYEVENTF_UNICODE | KEYEVENTF_KEYUP,
                            time=0,
                            dwExtraInfo=GetMessageExtraInfo(),
                        )
                    ),
                ),
            )
            pointer = ctypes.byref(inputs[0])
            SendInput(len(inputs), pointer, ctypes.sizeof(INPUT))
            time.sleep(0.01)

        return True
    except Exception as exc:
        logger.debug("Windows 传统输入失败: %s", exc)
        return False

