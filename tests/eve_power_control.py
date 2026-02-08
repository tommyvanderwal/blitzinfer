#!/usr/bin/env python3
"""
BLE Power Control for Eve Energy 20EBO8301

Controls an Eve Energy smart plug via Bluetooth Low Energy to power cycle
a crashed remote machine.

Device: Eve Energy 20EBO8301 (firmware 3.5.0, board 1.1)

IMPORTANT: Eve Energy with HomeKit
-------------------------------------
Eve Energy devices paired to HomeKit use encrypted HAP (HomeKit Accessory Protocol)
over BLE. This means:
1. You cannot control them with simple BLE GATT writes
2. You need the HomeKit pairing keys (stored in your Apple device)
3. Options for control:
   a) Use the Eve app directly
   b) Use Homebridge with homebridge-eve-energy plugin
   c) Factory reset the device and use it without HomeKit (loses HomeKit features)
   d) Use aiohomekit library with the pairing keys (complex setup)

For simple power cycling without HomeKit:
- Consider a Tapo/Kasa smart plug (has direct local API)
- Or a Shelly plug (has REST API)
- Or a Tuya plug with tinytuya library

Usage:
    python eve_power_control.py scan          # Find Eve devices
    python eve_power_control.py status        # Get current power state
    python eve_power_control.py on            # Turn on
    python eve_power_control.py off           # Turn off
    python eve_power_control.py cycle [delay] # Power cycle (off, wait, on)
    python eve_power_control.py auto          # Monitor and auto-recover crashed host
"""

import asyncio
import argparse
import sys
import subprocess
import time
from datetime import datetime

# Add user site-packages for bleak
sys.path.insert(0, '/home/tommy/.local/lib/python3.12/site-packages')

try:
    from bleak import BleakClient, BleakScanner
    from bleak.exc import BleakError
except ImportError:
    print("ERROR: bleak not installed. Run: pip install bleak")
    sys.exit(1)

# Eve Energy uses HAP (HomeKit Accessory Protocol) over BLE
# The On characteristic is in the HAP service

# Eve Energy BLE UUIDs (HomeKit)
# HAP Service UUID
HAP_SERVICE_UUID = "00000001-0000-1000-8000-0026bb765291"  # HAP Pairing Service

# Eve Energy uses a proprietary service for power control
# These are common Eve Energy UUIDs discovered via scanning
EVE_ENERGY_SERVICE = "e863f001-079e-48ff-8f27-9c2605a29f52"  # Eve proprietary
EVE_ENERGY_CHAR_POWER = "e863f00a-079e-48ff-8f27-9c2605a29f52"  # On/Off

# Alternative: Standard HAP On characteristic
HAP_ON_CHAR = "00000025-0000-1000-8000-0026bb765291"

# Device address (will be discovered or can be set)
# Format: XX:XX:XX:XX:XX:XX
DEVICE_ADDRESS = None  # Set this after scanning, or pass via --address

# Host to monitor for auto-recovery
REMOTE_HOST = "192.168.2.90"


def log(msg: str):
    """Print timestamped log message."""
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {msg}")


async def scan_for_eve_devices(timeout: float = 10.0):
    """Scan for Eve Energy devices."""
    log(f"Scanning for BLE devices ({timeout}s)...")

    devices = await BleakScanner.discover(timeout=timeout, return_adv=True)

    eve_devices = []
    for device, adv_data in devices.values():
        name = adv_data.local_name or device.name or "Unknown"

        # Eve devices typically advertise as "Eve Energy XXXX" or similar
        if "eve" in name.lower() or "energy" in name.lower():
            eve_devices.append((device, adv_data))
            log(f"  FOUND EVE: {name} @ {device.address}")
            log(f"    RSSI: {adv_data.rssi} dBm")
            if adv_data.service_uuids:
                log(f"    Services: {adv_data.service_uuids}")
        else:
            # Also show other devices for debugging
            if adv_data.rssi > -70:  # Only strong signals
                log(f"  Other: {name} @ {device.address} (RSSI: {adv_data.rssi})")

    if not eve_devices:
        log("No Eve devices found. Make sure:")
        log("  1. Bluetooth is enabled on this machine")
        log("  2. Eve Energy is powered and in range")
        log("  3. Eve Energy may need to be unpaired from other devices first")

        # Show all devices found
        log(f"\nAll devices found ({len(devices)}):")
        for device, adv_data in devices.values():
            name = adv_data.local_name or device.name or "Unknown"
            log(f"  {name} @ {device.address} (RSSI: {adv_data.rssi})")

    return eve_devices


async def discover_services(address: str):
    """Connect and discover all services/characteristics."""
    log(f"Connecting to {address}...")

    async with BleakClient(address) as client:
        log(f"Connected: {client.is_connected}")

        log("\nServices and Characteristics:")
        for service in client.services:
            log(f"\n  Service: {service.uuid}")
            if service.description:
                log(f"    Description: {service.description}")

            for char in service.characteristics:
                props = ", ".join(char.properties)
                log(f"    Char: {char.uuid}")
                log(f"      Properties: {props}")
                if char.description:
                    log(f"      Description: {char.description}")

                # Try to read if readable
                if "read" in char.properties:
                    try:
                        value = await client.read_gatt_char(char.uuid)
                        log(f"      Value: {value.hex()} ({list(value)})")
                    except Exception as e:
                        log(f"      Read error: {e}")


async def get_power_state(address: str) -> bool | None:
    """Get current power state of Eve Energy."""
    log(f"Getting power state from {address}...")

    try:
        async with BleakClient(address, timeout=10.0) as client:
            if not client.is_connected:
                log("Failed to connect")
                return None

            # Try Eve proprietary characteristic first
            try:
                value = await client.read_gatt_char(EVE_ENERGY_CHAR_POWER)
                # Usually 0x01 = on, 0x00 = off
                state = value[0] if value else None
                log(f"Power state: {'ON' if state else 'OFF'} (raw: {value.hex()})")
                return bool(state)
            except Exception as e:
                log(f"Eve char failed: {e}")

            # Try HAP On characteristic
            try:
                value = await client.read_gatt_char(HAP_ON_CHAR)
                state = value[0] if value else None
                log(f"Power state (HAP): {'ON' if state else 'OFF'} (raw: {value.hex()})")
                return bool(state)
            except Exception as e:
                log(f"HAP char failed: {e}")

            return None

    except BleakError as e:
        log(f"BLE Error: {e}")
        return None


async def set_power_state(address: str, on: bool) -> bool:
    """Set power state of Eve Energy."""
    state_str = "ON" if on else "OFF"
    log(f"Setting power {state_str} on {address}...")

    try:
        async with BleakClient(address, timeout=10.0) as client:
            if not client.is_connected:
                log("Failed to connect")
                return False

            value = bytes([0x01 if on else 0x00])

            # Try Eve proprietary characteristic
            try:
                await client.write_gatt_char(EVE_ENERGY_CHAR_POWER, value)
                log(f"Power set to {state_str} via Eve char")
                return True
            except Exception as e:
                log(f"Eve char write failed: {e}")

            # Try HAP On characteristic
            try:
                await client.write_gatt_char(HAP_ON_CHAR, value)
                log(f"Power set to {state_str} via HAP char")
                return True
            except Exception as e:
                log(f"HAP char write failed: {e}")

            log("Failed to set power state - no writable characteristic found")
            return False

    except BleakError as e:
        log(f"BLE Error: {e}")
        return False


async def power_cycle(address: str, delay: float = 10.0) -> bool:
    """Power cycle: off, wait, on."""
    log(f"Power cycling with {delay}s delay...")

    if not await set_power_state(address, False):
        log("Failed to turn off")
        return False

    log(f"Waiting {delay}s...")
    await asyncio.sleep(delay)

    if not await set_power_state(address, True):
        log("Failed to turn on")
        return False

    log("Power cycle complete")
    return True


def check_host_alive(host: str, timeout: float = 5.0) -> bool:
    """Check if remote host responds to SSH."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
             f"tommy@{host}", "echo ok"],
            capture_output=True,
            timeout=timeout + 2
        )
        return result.returncode == 0 and b"ok" in result.stdout
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


async def auto_recovery_monitor(address: str, check_interval: float = 30.0,
                                 recovery_delay: float = 10.0):
    """Monitor remote host and auto-recover on crash."""
    log(f"Starting auto-recovery monitor for {REMOTE_HOST}")
    log(f"  Check interval: {check_interval}s")
    log(f"  Recovery delay: {recovery_delay}s")
    log(f"  Eve Energy: {address}")
    log("Press Ctrl+C to stop\n")

    consecutive_failures = 0
    last_state = True  # Assume host is up initially

    while True:
        try:
            alive = check_host_alive(REMOTE_HOST)

            if alive:
                if not last_state:
                    log(f"HOST RECOVERED - {REMOTE_HOST} is back online")
                else:
                    log(f"Host OK - {REMOTE_HOST}")
                consecutive_failures = 0
                last_state = True
            else:
                consecutive_failures += 1
                log(f"Host DOWN - {REMOTE_HOST} (failures: {consecutive_failures})")

                # After 2 consecutive failures, trigger recovery
                if consecutive_failures >= 2 and last_state:
                    log("TRIGGERING POWER CYCLE RECOVERY")
                    last_state = False

                    success = await power_cycle(address, recovery_delay)
                    if success:
                        log(f"Power cycle complete. Waiting for host to boot...")
                        # Give host time to boot (typically 60-120s)
                        await asyncio.sleep(90)
                    else:
                        log("Power cycle failed!")

            await asyncio.sleep(check_interval)

        except KeyboardInterrupt:
            log("\nStopping monitor")
            break
        except Exception as e:
            log(f"Monitor error: {e}")
            await asyncio.sleep(check_interval)


async def main():
    parser = argparse.ArgumentParser(description="Eve Energy BLE Power Control")
    parser.add_argument("command", choices=["scan", "discover", "status", "on", "off", "cycle", "auto"],
                        help="Command to execute")
    parser.add_argument("--address", "-a", type=str,
                        help="BLE device address (XX:XX:XX:XX:XX:XX)")
    parser.add_argument("--delay", "-d", type=float, default=10.0,
                        help="Power cycle delay in seconds (default: 10)")
    parser.add_argument("--interval", "-i", type=float, default=30.0,
                        help="Auto-monitor check interval (default: 30)")

    args = parser.parse_args()

    if args.command == "scan":
        devices = await scan_for_eve_devices()
        if devices:
            log(f"\nFound {len(devices)} Eve device(s)")
            log("Use --address with one of the addresses above")
        return

    # All other commands need an address
    address = args.address or DEVICE_ADDRESS
    if not address:
        log("ERROR: Device address required. Run 'scan' first or use --address")
        log("Example: python eve_power_control.py --address AA:BB:CC:DD:EE:FF status")
        sys.exit(1)

    if args.command == "discover":
        await discover_services(address)
    elif args.command == "status":
        await get_power_state(address)
    elif args.command == "on":
        await set_power_state(address, True)
    elif args.command == "off":
        await set_power_state(address, False)
    elif args.command == "cycle":
        await power_cycle(address, args.delay)
    elif args.command == "auto":
        await auto_recovery_monitor(address, args.interval, args.delay)


if __name__ == "__main__":
    asyncio.run(main())
