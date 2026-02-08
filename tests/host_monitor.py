#!/usr/bin/env python3
"""
Simple host monitor that alerts when the remote system crashes.

Usage:
    python host_monitor.py                    # Monitor with default settings
    python host_monitor.py --interval 30      # Check every 30 seconds
    python host_monitor.py --sound            # Play sound alert on crash

For Eve Energy control, you'll need to use the Eve app or Homebridge.
This script can be extended to call external power control APIs.
"""

import subprocess
import sys
import time
import argparse
from datetime import datetime


def log(msg: str):
    """Print timestamped log message."""
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {msg}", flush=True)


def check_host_alive(host: str, timeout: float = 5.0) -> bool:
    """Check if remote host responds to ping."""
    try:
        result = subprocess.run(
            ["ping", "-c", "1", "-W", str(int(timeout)), host],
            capture_output=True,
            timeout=timeout + 2
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def check_ssh_alive(host: str, user: str = "tommy", timeout: float = 5.0) -> bool:
    """Check if remote host responds to SSH."""
    try:
        result = subprocess.run(
            ["ssh", "-o", f"ConnectTimeout={int(timeout)}", "-o", "BatchMode=yes",
             f"{user}@{host}", "echo ok"],
            capture_output=True,
            timeout=timeout + 5
        )
        return result.returncode == 0 and b"ok" in result.stdout
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def play_alert_sound():
    """Play an alert sound using system beep or paplay."""
    try:
        # Try paplay first (PulseAudio)
        subprocess.run(["paplay", "/usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga"],
                      capture_output=True, timeout=5)
    except Exception:
        try:
            # Fall back to beep
            subprocess.run(["beep", "-f", "1000", "-l", "500"], capture_output=True, timeout=2)
        except Exception:
            # Last resort: terminal bell
            print("\a\a\a", flush=True)


def notify_desktop(title: str, message: str):
    """Send desktop notification."""
    try:
        subprocess.run(["notify-send", "-u", "critical", title, message],
                      capture_output=True, timeout=5)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Monitor remote host for crashes")
    parser.add_argument("--host", "-H", type=str, default="192.168.2.90",
                       help="Host to monitor (default: 192.168.2.90)")
    parser.add_argument("--user", "-u", type=str, default="tommy",
                       help="SSH user (default: tommy)")
    parser.add_argument("--interval", "-i", type=float, default=10.0,
                       help="Check interval in seconds (default: 10)")
    parser.add_argument("--sound", "-s", action="store_true",
                       help="Play sound alert on crash")
    parser.add_argument("--use-ssh", action="store_true",
                       help="Use SSH check instead of ping (slower but more accurate)")

    args = parser.parse_args()

    log(f"Starting host monitor for {args.host}")
    log(f"  Check interval: {args.interval}s")
    log(f"  Method: {'SSH' if args.use_ssh else 'Ping'}")
    log(f"  Sound alerts: {'ON' if args.sound else 'OFF'}")
    log("Press Ctrl+C to stop\n")

    consecutive_failures = 0
    was_down = False

    while True:
        try:
            if args.use_ssh:
                alive = check_ssh_alive(args.host, args.user)
            else:
                alive = check_host_alive(args.host)

            if alive:
                if was_down:
                    log(f"HOST RECOVERED - {args.host} is back online!")
                    notify_desktop("Host Recovered", f"{args.host} is back online")
                    if args.sound:
                        play_alert_sound()
                else:
                    # Print status every 10th check to show script is running
                    if consecutive_failures == 0:
                        log(f"Host OK - {args.host}")

                consecutive_failures = 0
                was_down = False
            else:
                consecutive_failures += 1
                log(f"Host DOWN - {args.host} (check #{consecutive_failures})")

                # After 2 consecutive failures, declare crash
                if consecutive_failures >= 2 and not was_down:
                    log("=" * 60)
                    log("!!! HOST CRASHED !!!")
                    log(f"    {args.host} is not responding")
                    log("    Manual power cycle required")
                    log("=" * 60)
                    notify_desktop("HOST CRASHED!", f"{args.host} needs power cycle")
                    if args.sound:
                        play_alert_sound()
                    was_down = True

            time.sleep(args.interval)

        except KeyboardInterrupt:
            log("\nStopping monitor")
            break
        except Exception as e:
            log(f"Monitor error: {e}")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
