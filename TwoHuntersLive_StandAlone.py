import sys
import time
import json
import logging
import threading
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from datetime import datetime, timedelta, time as dt_time
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import click
import yaml
import MetaTrader5 as mt5


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Settings:
    """Singleton configuration manager — reads TwoHuntersLive_StandAlone.yaml."""

    _instance   = None
    _config: Optional[Dict[str, Any]] = None
    _config_path: Optional[Path]      = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self.load_config()

    # ------------------------------------------------------------------
    # Load / validate / save
    # ------------------------------------------------------------------

    def load_config(self, config_path: Optional[str] = None):
        if config_path is None:
            if self._config_path:
                config_path = self._config_path
            else:
                base = Path(sys._MEIPASS) if getattr(sys, "frozen", False) else Path(__file__).parent
                config_path = base / "TwoHuntersLive_StandAlone.yaml"
        self._config_path = Path(config_path)
        try:
            with open(self._config_path, "r", encoding="utf-8") as fh:
                self._config = yaml.safe_load(fh)
            self._validate_config()
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as exc:
            raise ValueError(f"Error parsing configuration file: {exc}")

    def save_config(self):
        """Write current in-memory config back to the YAML file (preserves all keys)."""
        if self._config is None or self._config_path is None:
            return
        try:
            with open(self._config_path, "w", encoding="utf-8") as fh:
                yaml.dump(self._config, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
        except Exception as exc:
            get_logger().error("Failed to save config: %s", exc)

    def _validate_config(self):
        required = [
            "system.name", "symbols", "mt5.magic_number",
            "risk_percent", "mbox_time.start", "mbox_time.end", "sessions.main.end",
        ]
        for field in required:
            if self.get(field) is None:
                raise ValueError(f"Required configuration field missing: {field}")
        if not isinstance(self.get("symbols", []), list):
            raise ValueError("symbols must be a list")
        risk = self.get("risk_percent", 0)
        if not 0 < risk <= 0.1:
            raise ValueError("risk_percent must be between 0 and 0.1 (exclusive)")

    # ------------------------------------------------------------------
    # Generic accessors
    # ------------------------------------------------------------------

    def get(self, key_path: str, default: Any = None) -> Any:
        if self._config is None:
            self.load_config()
        value = self._config
        try:
            for key in key_path.split("."):
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key_path: str, value: Any):
        if self._config is None:
            self.load_config()
        keys   = key_path.split(".")
        config = self._config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value

    # ------------------------------------------------------------------
    # admin_commands properties
    # ------------------------------------------------------------------

    @property
    def admin_update(self) -> bool:
        return bool(self.get("admin_commands.update", False))

    @property
    def admin_start(self) -> bool:
        return bool(self.get("admin_commands.start", False))

    @property
    def admin_stop(self) -> bool:
        return bool(self.get("admin_commands.stop", False))

    @property
    def admin_reset(self) -> bool:
        return bool(self.get("admin_commands.reset", False))

    @property
    def admin_pause(self) -> bool:
        return bool(self.get("admin_commands.pause", False))

    @property
    def admin_resume(self) -> bool:
        return bool(self.get("admin_commands.resume", False))

    @property
    def admin_pause_symbol(self) -> Optional[str]:
        v = self.get("admin_commands.pause_symbol", None)
        return str(v) if v else None

    @property
    def admin_resume_symbol(self) -> Optional[str]:
        v = self.get("admin_commands.resume_symbol", None)
        return str(v) if v else None

    @property
    def admin_force_order(self) -> bool:
        return bool(self.get("admin_commands.force_order", False))

    # ------------------------------------------------------------------
    # Typed convenience properties
    # ------------------------------------------------------------------

    @property
    def system_name(self) -> str:
        return self.get("system.name", "TwoHuntersLive_StandAlone")

    @property
    def system_version(self) -> str:
        return self.get("system.version", "1.0.0")

    @property
    def path_logs(self) -> str:
        return self.get("paths.logs", "reports/logs")

    @property
    def path_fvgs(self) -> str:
        return self.get("paths.fvgs", "reports/fvgs")

    @property
    def path_signals(self) -> str:
        return self.get("paths.signals", "reports/signals")

    @property
    def mt5_magic_number(self) -> int:
        return self.get("mt5.magic_number", 842451994)

    @property
    def mt5_deviation(self) -> int:
        return self.get("mt5.deviation", 10)

    @property
    def mt5_reconnect_attempts(self) -> int:
        return self.get("mt5.reconnect_attempts", 6)

    @property
    def mt5_reconnect_delay(self) -> int:
        return self.get("mt5.reconnect_delay", 5)

    @property
    def symbols(self) -> List[str]:
        return self.get("symbols", ["EURUSD.", "GBPUSD."])

    @property
    def risk_percent(self) -> float:
        return self.get("risk_percent", 0.01)

    @property
    def timeframe(self) -> str:
        return self.get("timeframe", "M1")

    @property
    def num_hunt_main(self) -> int:
        return self.get("num_hunt_main", 2)

    @property
    def should_recover(self) -> bool:
        return self.get("should_recover", False)

    @property
    def orderblock_fvg_pip_size(self) -> Dict[str, float]:
        return self.get("orderblock_fvg_pip_size", {"min": 3.5, "max": 5.0})

    @property
    def sl_ratio(self) -> float:
        return self.get("ratios.stop_loss", 1.0)

    @property
    def tp_ratio(self) -> float:
        return self.get("ratios.take_profit", 3.0)

    @property
    def margin_pips(self) -> float:
        return self.get("margin_pips", 0.01)

    @property
    def work_interval(self) -> int:
        return self.get("work_interval", 1)

    @property
    def skip_minutes(self) -> int:
        return self.get("skip_minutes", 60)

    @property
    def retry_order_minutes(self) -> int:
        return self.get("retry_order_minutes", 5)

    @property
    def mbox_start(self) -> str:
        return self.get("mbox_time.start", "04:30")

    @property
    def mbox_end(self) -> str:
        return self.get("mbox_time.end", "12:29")

    @property
    def session_main_start(self) -> str:
        return self.get("sessions.main.start", "12:29")

    @property
    def session_main_end(self) -> str:
        return self.get("sessions.main.end", "22:00")

    @property
    def session_london(self) -> Dict[str, str]:
        return self.get("sessions.london", {"start": "14:00", "end": "22:00"})

    @property
    def session_newyork(self) -> Dict[str, str]:
        return self.get("sessions.newyork", {"start": "20:00", "end": "02:29"})

    @property
    def use_time_flag(self) -> bool:
        return self.get("use_time_flag", True)

    @property
    def time_flag_hour(self) -> str:
        return self.get("time_flag_hour", "10:30")

    @property
    def fvg_timeframe(self) -> str:
        return self.get("fvg_timeframe", "H4")

    @property
    def lookup_days(self) -> int:
        return self.get("lookup_days", 5)

    @property
    def log_level(self) -> str:
        return self.get("logging.level", "INFO")

    @property
    def log_to_file(self) -> bool:
        return self.get("logging.log_to_file", True)

    @property
    def log_to_console(self) -> bool:
        return self.get("logging.log_to_console", True)


settings = Settings()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class _SymbolFilter(logging.Filter):
    """Prepend [SYMBOL] to every log record that passes through this logger."""
    def __init__(self, symbol: str):
        super().__init__()
        self._tag = f"[{symbol.rstrip('.').upper()}]" if symbol else ""

    def filter(self, record: logging.LogRecord) -> bool:
        if self._tag and not record.getMessage().startswith(self._tag):
            record.msg = f"{self._tag} {record.msg}"
        return True


_ROOT_LOGGER_NAME = "TwoHunters"
_logger_ready     = False


def _build_logger(verbose: bool = False, quiet: bool = False) -> None:
    """
    Configure the root TwoHunters logger.

    Console level:
        quiet   → WARNING
        verbose → DEBUG
        default → INFO

    File level: always DEBUG (daily rotating, reports/logs/YYYY-MM-DD.log).

    Safe to call multiple times — clears existing handlers on the root logger
    and rebuilds them, allowing verbose/quiet to be applied after early
    module-level calls have already set up a default handler.
    """
    global _logger_ready

    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.handlers.clear()          # remove any previously added handlers
    root.setLevel(logging.DEBUG)
    root.propagate = False

    fmt = logging.Formatter(
        fmt     = "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt = "%H:%M:%S",
    )

    # ── File handler ──────────────────────────────────────────────────
    if settings.log_to_file:
        log_dir = Path(settings.path_logs)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / (datetime.now().strftime("%Y-%m-%d") + ".log")
        fh = TimedRotatingFileHandler(
            filename    = str(log_path),
            when        = "midnight",
            interval    = 1,
            backupCount = 30,
            encoding    = "utf-8",
            utc         = False,
        )
        fh.suffix    = "%Y-%m-%d"
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    # ── Console handler ───────────────────────────────────────────────
    if settings.log_to_console:
        ch = logging.StreamHandler(sys.stdout)
        if quiet:
            ch.setLevel(logging.WARNING)
        elif verbose:
            ch.setLevel(logging.DEBUG)
        else:
            ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        root.addHandler(ch)

    _logger_ready = True


def get_logger(symbol: str = "") -> logging.Logger:
    """
    Return a child logger for a specific symbol.
    Ensures the root logger always has at least a default INFO handler —
    covers calls that happen before two_hunters_cli() runs _build_logger
    (e.g. AdminController.__init__ at module level, place_order in threads).
    Safe to call multiple times — symbol filters are never duplicated.
    """
    if not _logger_ready:
        _build_logger()   # set up default INFO handler on first call

    name   = f"{_ROOT_LOGGER_NAME}.{symbol.rstrip('.').upper()}" if symbol else _ROOT_LOGGER_NAME
    logger = logging.getLogger(name)
    if not logger.filters:
        logger.addFilter(_SymbolFilter(symbol))
    return logger


# ---------------------------------------------------------------------------
# Admin controller
# ---------------------------------------------------------------------------

class AdminController:
    """
    Polls admin_commands in the YAML file and applies them at the top of
    every main-loop iteration.  Thread-safe via a single lock.

    Priority (evaluated in this order each poll):
        1. stop
        2. reset
        3. update
        4. start        (no-op — process is already running)
        5. pause  /  pause_symbol
        6. resume /  resume_symbol
        7. force_order  (persistent — stays until operator sets it back to false)
    """

    def __init__(self):
        self._lock: threading.Lock         = threading.Lock()
        self._log                          = get_logger()

        # ── Pause state ───────────────────────────────────────────────
        self._global_paused: bool          = False
        self._paused_symbols: Set[str]     = set()
        self._resumed_symbols: Set[str]    = set()

        # Signals generated while paused, waiting to be placed on resume
        # Key: normalised symbol string (rstrip(".").upper())
        self.pending_signals: Dict[str, Any] = {}

        # ── force_order ───────────────────────────────────────────────
        self._force_order: bool            = False

        # ── reset helper ─────────────────────────────────────────────
        self._reset_requested: bool        = False

        # ── halt helper ──────────────────────────────────────────────
        self._halt_requested: bool         = False

    # ------------------------------------------------------------------
    # Public queries  (called from worker threads — lock-protected)
    # ------------------------------------------------------------------

    def is_paused(self, symbol: str) -> bool:
        """Returns True if the given symbol must not place new orders right now."""
        with self._lock:
            return self._global_paused or symbol in self._paused_symbols

    @property
    def force_order(self) -> bool:
        with self._lock:
            return self._force_order

    @property
    def reset_requested(self) -> bool:
        with self._lock:
            return self._reset_requested

    def clear_reset(self):
        with self._lock:
            self._reset_requested = False

    # ------------------------------------------------------------------
    # Main poll — called at top of live() main loop
    # ------------------------------------------------------------------

    def poll_and_apply(
        self,
        stop_event:      threading.Event,
        thread_registry: Dict[str, Dict],
        mt5_conn:        Any,
    ) -> None:
        """
        Re-read admin_commands from YAML, act on any set flags, then
        write flags back to false/null so they are not re-triggered.
        """
        # Re-read YAML from disk to pick up operator edits
        try:
            if settings._config_path and settings._config_path.exists():
                with open(settings._config_path, "r", encoding="utf-8") as fh:
                    fresh = yaml.safe_load(fh)
                cmds = fresh.get("admin_commands", {}) if fresh else {}
            else:
                return
        except Exception as exc:
            self._log.warning("Admin poll — could not read YAML: %s", exc)
            return

        dirty = False   # set to True whenever we modify a flag in cmds

        # ── 0. halt ───────────────────────────────────────────────────
        if cmds.get("halt"):
            self._log.warning("ADMIN ▶ halt — halting live trading")
            self._halt_requested = True
            dirty = True
        elif self._halt_requested and not cmds.get("halt"):
            self._log.warning("ADMIN ▶ halt cleared — resuming live trading")
            self._halt_requested = False
            dirty = True

        # ── 1. stop ───────────────────────────────────────────────────
        if cmds.get("stop", False):
            self._log.warning("ADMIN ▶ stop — shutting down live trading")
            stop_event.set()
            cmds["stop"] = False
            dirty = True

        # ── 2. reset ──────────────────────────────────────────────────
        if cmds.get("reset", False):
            self._log.warning("ADMIN ▶ reset — stopping all threads, reloading context")
            stop_event.set()
            with self._lock:
                self._reset_requested = True
            cmds["reset"] = False
            dirty = True

        # ── 3. update ─────────────────────────────────────────────────
        if cmds.get("update", False):
            self._log.info("ADMIN ▶ update — reloading configuration from YAML")
            try:
                settings.load_config()
                self._log.info("Configuration reloaded successfully")
            except Exception as exc:
                self._log.error("Configuration reload failed: %s", exc)
            cmds["update"] = False
            dirty = True

        # ── 4. say_hello (no-op: process is running) ──────────────
        if cmds.get("say_hello", False):
            self._log.info("ADMIN says hello — system is running")
            cmds["say_hello"] = False
            dirty = True

        # ── 5a. pause (all symbols) ───────────────────────────────────
        if cmds.get("pause") and not self._global_paused:
            with self._lock:
                self._global_paused = True
            self._log.info("ADMIN ▶ pause — order placement suspended for ALL symbols")
            dirty = True
        
        elif not cmds.get("pause") and self._global_paused:
            with self._lock:
                self._global_paused = False
            self._log.info("ADMIN ▶ resume — order placement resumed for ALL symbols")
            dirty = True

        # ── 5b. pause_symbol ─────────────────────────────────────────
        ps = cmds.get("pause_symbol", None)
        if ps:
            norm = str(ps).strip()
            with self._lock:
                self._paused_symbols.add(norm)
            self._log.info("ADMIN ▶ pause_symbol — order placement suspended for %s", norm)
            cmds["pause_symbol"] = None
            dirty = True

        # ── 6 resume_symbol ─────────────────────────────────────────
        rs = cmds.get("resume_symbol", None)
        if rs:
            norm = str(rs).strip()
            with self._lock:
                self._paused_symbols.discard(norm)
            self._log.info("ADMIN ▶ resume_symbol — order placement resumed for %s", norm)
            cmds["resume_symbol"] = None
            dirty = True

        # ── 7. force_order (persistent — stays set until operator clears) ──
        fo = cmds.get("force_order")
        with self._lock:
            if fo != self._force_order:
                dirty = True
                self._force_order = fo
                if self._force_order:
                    self._log.info(
                        "ADMIN ▶ force_order ENABLED — orders with Invalid Stops "
                        "will be retried without SL/TP"
                    )
                else:
                    self._log.info("ADMIN ▶ force_order DISABLED")

        # Write modified flags back to disk only if something changed
        if dirty:
            try:
                fresh["admin_commands"] = cmds
                with open(settings._config_path, "w", encoding="utf-8") as fh:
                    yaml.dump(fresh, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
            except Exception as exc:
                self._log.error("Failed to write admin flags back to YAML: %s", exc)


# Module-level singleton — created before live() runs
admin = AdminController()

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SignalAction(Enum):
    BUY  = "BUY"
    SELL = "SELL"


class SignalOutcome(Enum):
    WIN          = "win"
    LOSS         = "loss"
    PENDING      = "pending"
    FORCE_STOPED = "force_stoped"


class SignalType(Enum):
    MAIN     = "main"
    RECOVERY = "recovery"


# ---------------------------------------------------------------------------
# Bar
# ---------------------------------------------------------------------------

class Bar:
    def __init__(
        self,
        timestamp: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: int = 0,
    ):
        self.timestamp = timestamp
        self.open      = open_price
        self.high      = high
        self.low       = low
        self.close     = close
        self.volume    = volume
        self._calculate_attributes()

    def _calculate_attributes(self):
        self.body        = abs(self.close - self.open)
        self.range       = self.high - self.low
        self.upper_wick  = self.high - max(self.open, self.close)
        self.lower_wick  = min(self.open, self.close) - self.low
        self.is_bullish  = self.close > self.open
        self.is_bearish  = self.close < self.open
        self.is_doji     = self.close == self.open
        self._classify_candle()

    def _classify_candle(self):
        self.is_weak      = False
        self.is_head_down = False
        self.is_head_up   = False

        if self.body == 0:
            self.is_weak = True
            self.is_doji = True
        elif self.range > 0:
            if self.body / self.range < 0.3:
                self.is_weak = True
                if self.upper_wick >= 2 * self.lower_wick:
                    self.is_head_down = True
                elif self.lower_wick >= 2 * self.upper_wick:
                    self.is_head_up = True
                else:
                    self.is_doji = True
        else:
            self.range = 0.0000001


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------

class Signal:
    def __init__(
        self,
        action: SignalAction,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        symbol: str,
        timestamp: datetime,
        signal_type: SignalType = SignalType.MAIN,
        take_profit_pips: Optional[float] = None,
        stop_loss_pips: Optional[float] = None,
        entry_lot: Optional[float] = None,
        gain: Optional[float] = None,
        ticket: Optional[int] = None,
    ):
        self.action      = action if isinstance(action, SignalAction) else SignalAction(action)
        self.symbol      = symbol
        self.timestamp   = timestamp
        self.signal_type = signal_type

        self.entry_price  = entry_price
        self.stop_loss    = stop_loss
        self.take_profit  = take_profit

        self.initial_entry_price = entry_price
        self.initial_stop_loss   = stop_loss
        self.initial_take_profit = take_profit

        self.entry_lot        = entry_lot
        self.stop_loss_pips   = stop_loss_pips
        self.take_profit_pips = take_profit_pips

        self.outcome: Optional[SignalOutcome] = SignalOutcome.PENDING
        self.outcome_timestamp: Optional[datetime] = None
        self.exit_pips:  Optional[float] = None
        self.exit_price: Optional[float] = None
        self.gain = gain

        self.ticket     = ticket
        self.commission = None

        self.trend      = None
        self.fake_CHoCH = None
        self.time_flag  = None

        self.sl_adjusted_count = 0

    @property
    def is_main(self) -> bool:
        return self.signal_type == SignalType.MAIN

    @property
    def is_buy(self) -> bool:
        return self.action == SignalAction.BUY

    @property
    def is_sell(self) -> bool:
        return self.action == SignalAction.SELL

    @property
    def is_pending(self) -> bool:
        return self.outcome is None or self.outcome == SignalOutcome.PENDING

    @property
    def is_completed(self) -> bool:
        return self.outcome in (SignalOutcome.WIN, SignalOutcome.LOSS, SignalOutcome.FORCE_STOPED)

    def to_dict(self) -> dict:
        return {
            "action":               self.action.value,
            "symbol":               self.symbol,
            "timestamp":            self.timestamp.isoformat(),
            "signal_type":          self.signal_type.value,
            "entry_price":          self.entry_price,
            "stop_loss":            self.stop_loss,
            "take_profit":          self.take_profit,
            "initial_entry_price":  self.initial_entry_price,
            "initial_stop_loss":    self.initial_stop_loss,
            "initial_take_profit":  self.initial_take_profit,
            "entry_lot":            self.entry_lot,
            "stop_loss_pips":       self.stop_loss_pips,
            "take_profit_pips":     self.take_profit_pips,
            "outcome":              self.outcome.value if self.outcome else None,
            "outcome_timestamp":    self.outcome_timestamp.isoformat() if self.outcome_timestamp else None,
            "exit_pips":            self.exit_pips,
            "exit_price":           self.exit_price,
            "gain":                 self.gain,
            "ticket":               self.ticket,
            "commission":           self.commission,
            "trend":                self.trend,
            "fake_CHoCH":           self.fake_CHoCH,
            "time_flag":            self.time_flag,
            "sl_adjusted_count":    self.sl_adjusted_count,
            "is_complete":          self.is_completed,
            "is_pending":           self.is_pending,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Signal":
        signal = cls(
            action       = SignalAction(data.get("action", "BUY")),
            entry_price  = float(data.get("entry_price",  0)),
            stop_loss    = float(data.get("stop_loss",    0)),
            take_profit  = float(data.get("take_profit",  0)),
            symbol       = data.get("symbol", ""),
            timestamp    = datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            signal_type  = SignalType(data.get("signal_type", "main")),
            take_profit_pips = float(data["take_profit_pips"]) if data.get("take_profit_pips") else None,
            stop_loss_pips   = float(data["stop_loss_pips"])   if data.get("stop_loss_pips")   else None,
            entry_lot        = float(data["entry_lot"])         if data.get("entry_lot")         else None,
            gain             = float(data["gain"])              if data.get("gain")              else None,
            ticket           = int(data["ticket"])              if data.get("ticket")            else None,
        )
        signal.initial_entry_price = float(data.get("initial_entry_price", signal.entry_price))
        signal.initial_stop_loss   = float(data.get("initial_stop_loss",   signal.stop_loss))
        signal.initial_take_profit = float(data.get("initial_take_profit", signal.take_profit))

        if data.get("outcome"):
            signal.outcome = SignalOutcome(data["outcome"])
        if data.get("outcome_timestamp"):
            signal.outcome_timestamp = datetime.fromisoformat(data["outcome_timestamp"])

        signal.exit_pips         = float(data["exit_pips"])   if data.get("exit_pips")   else None
        signal.exit_price        = float(data["exit_price"])  if data.get("exit_price")  else None
        signal.commission        = float(data["commission"])  if data.get("commission")  else None
        signal.trend             = data.get("trend")
        signal.fake_CHoCH        = data.get("fake_CHoCH")
        signal.time_flag         = data.get("time_flag")
        signal.sl_adjusted_count = int(data.get("sl_adjusted_count", 0))
        return signal

    def __repr__(self):
        outcome_str = f" {self.outcome.value.upper()}" if self.outcome else " PENDING"
        gain_str    = f" Gain: ${self.gain:.2f}"       if self.gain is not None else ""
        lot_str     = f"{self.entry_lot:.2f}"           if self.entry_lot is not None else "?"
        return (
            f"Signal({self.timestamp.strftime('%Y-%m-%d %H:%M')} {self.symbol} "
            f"{self.action.value} @ {self.entry_price:.5f} "
            f"Lot:{lot_str}{outcome_str}{gain_str})"
        )

    def __str__(self):
        return self.__repr__()


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

class Budget:
    MIN_LOT_SIZE = 0.01

    _PIP_SIZES: Dict[str, float] = {
        "EURUSD": 0.0001, "GBPUSD": 0.0001, "AUDUSD": 0.0001, "NZDUSD": 0.0001,
        "USDCAD": 0.0001, "USDCHF": 0.0001, "EURGBP": 0.0001, "EURAUD": 0.0001,
        "EURNZD": 0.0001, "EURCHF": 0.0001, "EURCAD": 0.0001, "GBPAUD": 0.0001,
        "GBPNZD": 0.0001, "GBPCAD": 0.0001, "GBPCHF": 0.0001, "AUDNZD": 0.0001,
        "AUDCAD": 0.0001, "AUDCHF": 0.0001, "NZDCAD": 0.0001, "NZDCHF": 0.0001,
        "CADCHF": 0.0001,
        "USDJPY": 0.01,  "EURJPY": 0.01,  "GBPJPY": 0.01,  "AUDJPY": 0.01,
        "NZDJPY": 0.01,  "CADJPY": 0.01,  "CHFJPY": 0.01,
        "XAUUSD": 0.01,  "XAGUSD": 0.001, "WTIUSD": 0.01,  "UKOIL":  0.01,
        "US30":   1.0,   "US500":  0.1,   "NAS100": 0.1,
        "GER30":  1.0,   "UK100":  1.0,   "JPN225": 1.0,
    }

    _LOT_UNITS: Dict[str, int] = {
        "EURUSD": 100000, "GBPUSD": 100000, "AUDUSD": 100000, "NZDUSD": 100000,
        "USDCAD": 100000, "USDCHF": 100000, "EURGBP": 100000, "EURAUD": 100000,
        "EURNZD": 100000, "EURCHF": 100000, "EURCAD": 100000, "GBPAUD": 100000,
        "GBPNZD": 100000, "GBPCAD": 100000, "GBPCHF": 100000, "AUDNZD": 100000,
        "AUDCAD": 100000, "AUDCHF": 100000, "NZDCAD": 100000, "NZDCHF": 100000,
        "CADCHF": 100000,
        "USDJPY": 1000,   "EURJPY": 1000,   "GBPJPY": 1000,   "AUDJPY": 1000,
        "NZDJPY": 1000,   "CADJPY": 1000,   "CHFJPY": 1000,
        "XAUUSD": 100000, "XAGUSD": 500,    "WTIUSD": 1000,   "UKOIL":  1000,
        "US30":   1000,   "US500":  100,    "NAS100": 100,
        "GER30":  1000,   "UK100":  1000,   "JPN225": 1000,
    }

    def __init__(self, initial_balance: float, initial_risk_percent: Optional[float] = None):
        self.initial_balance      = initial_balance
        self.current_balance      = initial_balance
        self.initial_risk_percent = initial_risk_percent or settings.risk_percent
        self.current_risk_percent = self.initial_risk_percent
        self.pip_size: Optional[float] = None
        self.lot_size: Optional[int]   = None

    def reset(self, balance: Optional[float] = None):
        self.current_balance      = balance if balance is not None else self.initial_balance
        self.current_risk_percent = self.initial_risk_percent

    def update_balance(self, new_balance: float):
        self.current_balance = new_balance

    def pips_from_diff(self, price_diff: float) -> float:
        return abs(price_diff) / self.pip_size

    def risk_amount(self) -> float:
        return self.current_risk_percent * self.current_balance

    def calculate_pip_size(self, symbol: str) -> Optional[float]:
        if symbol is None:
            return None
        self.pip_size = self._PIP_SIZES.get(symbol.rstrip(".").upper())
        return self.pip_size

    def calculate_lot_size(self, symbol: str) -> Optional[int]:
        if symbol is None:
            return None
        self.lot_size = self._LOT_UNITS.get(symbol.rstrip(".").upper())
        return self.lot_size

    def calculate_pip_value(self, symbol: str) -> float:
        return self.calculate_pip_size(symbol) * self.calculate_lot_size(symbol)

    def lots_from_diff(self, symbol: str, sl_distance: float) -> float:
        sl_pips   = self.pips_from_diff(sl_distance)
        if sl_pips <= 0:
            return self.MIN_LOT_SIZE
        pip_value = self.calculate_pip_value(symbol)
        lots      = self.risk_amount() / (sl_pips * pip_value) if pip_value else 0.0
        return max(round(lots, 2), self.MIN_LOT_SIZE)

    def calculate_gain_loss(
        self, symbol: str, entry_price: float, exit_price: float,
        lot_size: float, action: str,
    ) -> float:
        if action.upper() == "BUY":
            pips_moved = self.pips_from_diff(exit_price - entry_price)
            if exit_price < entry_price:
                pips_moved = -pips_moved
        else:
            pips_moved = self.pips_from_diff(entry_price - exit_price)
            if entry_price < exit_price:
                pips_moved = -pips_moved
        return pips_moved * self.calculate_pip_value(symbol) * lot_size

    def update_risk_percent(self, signal: "Signal"):
        self.current_risk_percent = self.initial_risk_percent
        if settings.use_time_flag and getattr(signal, "time_flag", False):
            self.current_risk_percent = self.initial_risk_percent / 2.0


# ---------------------------------------------------------------------------
# MT5Connection
# ---------------------------------------------------------------------------

class MT5Connection:
    def __init__(self):
        self.magic_number       = settings.mt5_magic_number
        self.deviation          = settings.mt5_deviation
        self.reconnect_attempts = settings.mt5_reconnect_attempts
        self.reconnect_delay    = settings.mt5_reconnect_delay
        self._connected         = False

    def initialize_connection(self) -> bool:
        log = get_logger()
        for attempt in range(self.reconnect_attempts):
            try:
                if mt5.initialize():
                    self._connected = True
                    log.debug("MT5 initialized on attempt %d", attempt + 1)
                    return True
            except Exception as exc:
                log.debug("MT5 init attempt %d failed: %s", attempt + 1, exc)
            if attempt < self.reconnect_attempts - 1:
                time.sleep(self.reconnect_delay)
        self._connected = False
        log.error("MT5 initialization failed after %d attempts", self.reconnect_attempts)
        return False

    def shutdown_connection(self):
        if self._connected:
            try:
                mt5.shutdown()
                self._connected = False
            except Exception:
                pass

    def ensure_connection(self) -> bool:
        if not self._connected:
            return self.initialize_connection()
        try:
            mt5.account_info()
            return True
        except Exception:
            self._connected = False
            get_logger().warning("MT5 connection lost — reconnecting")
            return self.initialize_connection()

    def get_account_info(self) -> Optional[object]:
        if not self.ensure_connection():
            return None
        try:
            return mt5.account_info()
        except Exception:
            return None

    def get_today_signals(self) -> List[Signal]:
        if not self.ensure_connection():
            return []
        try:
            from_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
            to_time   = datetime.now() + timedelta(days=1)
            deals     = mt5.history_deals_get(from_time, to_time)
            if not deals:
                return []

            signals: List[Signal]    = []
            processed_positions: set = set()

            for deal in deals:
                if (
                    hasattr(deal, "entry")
                    and deal.entry == mt5.DEAL_ENTRY_OUT
                    and deal.position_id not in processed_positions
                ):
                    processed_positions.add(deal.position_id)
                    entry_deal = next(
                        (d for d in deals
                         if d.position_id == deal.position_id and d.entry == mt5.DEAL_ENTRY_IN),
                        None,
                    )
                    if not entry_deal:
                        continue

                    action = SignalAction.BUY if entry_deal.type == mt5.ORDER_TYPE_BUY else SignalAction.SELL
                    sig = Signal(
                        action       = action,
                        entry_price  = entry_deal.price,
                        stop_loss    = 0.0,
                        take_profit  = 0.0,
                        symbol       = entry_deal.symbol,
                        timestamp    = datetime.fromtimestamp(entry_deal.time),
                        entry_lot    = entry_deal.volume,
                        gain         = deal.profit,
                        ticket       = entry_deal.order,
                    )
                    sig.outcome           = SignalOutcome.WIN if deal.profit >= 0 else SignalOutcome.LOSS
                    sig.outcome_timestamp = datetime.fromtimestamp(deal.time)
                    signals.append(sig)
            return signals
        except Exception:
            return []

    def place_order(self, signal: Signal, force: bool = False) -> bool:
        """
        Place an order for `signal`.

        force=True  →  if the broker returns TRADE_RETCODE_INVALID_STOPS,
                        retry immediately without SL and TP (market order only).
        """
        if not self.ensure_connection():
            return False
        log = get_logger(signal.symbol)
        try:
            _atmp = 0
            tick = mt5.symbol_info_tick(signal.symbol)
            while not tick:
                _atmp += 1
                if _atmp == 10:
                    log.warning("No tick data for %s after 10 attempts, keeping order attempt", signal.symbol)
                if _atmp == 50:
                    log.warning("No tick data for %s after 50 attempts, keeping order attempt", signal.symbol)
                if _atmp == 100:
                    log.warning("No tick data for %s after 100 attempts, aborting order attempt", signal.symbol)
                    return False
                tick = mt5.symbol_info_tick(signal.symbol)
                time.sleep(0.1)

            if signal.is_sell:
                order_type  = mt5.ORDER_TYPE_SELL
                price       = tick.bid
                stop_loss   = signal.stop_loss   if signal.stop_loss   > price else price + 0.00001
                take_profit = signal.take_profit if signal.take_profit < price else price - 0.00001
            else:
                order_type  = mt5.ORDER_TYPE_BUY
                price       = tick.ask
                stop_loss   = signal.stop_loss   if signal.stop_loss   < price else price - 0.00001
                take_profit = signal.take_profit if signal.take_profit > price else price + 0.00001

            request = {
                "action":    mt5.TRADE_ACTION_DEAL,
                "symbol":    signal.symbol,
                "volume":    signal.entry_lot,
                "type":      order_type,
                "price":     price,
                "sl":        stop_loss,
                "tp":        take_profit,
                "deviation": self.deviation,
                "magic":     self.magic_number,
                "comment":   "TwoHuntersLive-StandAlone",
                "type_time": mt5.ORDER_TIME_GTC,
            }
            result = mt5.order_send(request)

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                signal.ticket      = result.order
                signal.entry_price = price
                log.debug("Order accepted by broker (ticket %s)", result.order)
                return True

            # ── force_order fallback: retry without SL/TP ─────────────
            if result.retcode == mt5.TRADE_RETCODE_INVALID_STOPS and force:
                log.warning(
                    "Invalid Stops error — retrying WITHOUT SL/TP (force_order is enabled)"
                )
                request_no_sl = {k: v for k, v in request.items() if k not in ("sl", "tp")}
                result2 = mt5.order_send(request_no_sl)
                if result2.retcode == mt5.TRADE_RETCODE_DONE:
                    signal.ticket      = result2.order
                    signal.entry_price = price
                    log.warning(
                        "Order placed WITHOUT SL/TP (ticket %s) — manual stop management required",
                        result2.order,
                    )
                    return True
                log.error("force_order retry also failed: retcode=%s", result2.retcode)
                return False

            log.warning("Order rejected: retcode=%s", result.retcode)
            return False

        except Exception as exc:
            log.error("place_order exception: %s", exc)
            return False


# ---------------------------------------------------------------------------
# DataFetcher
# ---------------------------------------------------------------------------

class DataFetcher:
    _TF_MAP: Dict[str, int] = {
        "M1":  mt5.TIMEFRAME_M1,
        "M5":  mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1":  mt5.TIMEFRAME_H1,
        "H4":  mt5.TIMEFRAME_H4,
        "D1":  mt5.TIMEFRAME_D1,
    }

    def _ensure_connection(self):
        if not mt5.initialize():
            raise RuntimeError("Failed to initialize MetaTrader 5")

    def fetch_bars_from_mt5(
        self,
        start_dt: datetime,
        end_dt: datetime,
        symbol: str,
        timeframe: str = "M1",
    ) -> List[Bar]:
        self._ensure_connection()
        try:
            rates = mt5.copy_rates_range(symbol, self._TF_MAP.get(timeframe, mt5.TIMEFRAME_M1), start_dt, end_dt)
            if rates is None or len(rates) == 0:
                return []
            return [
                Bar(
                    timestamp  = datetime.fromtimestamp(int(r["time"])),
                    open_price = float(r["open"]),
                    high       = float(r["high"]),
                    low        = float(r["low"]),
                    close      = float(r["close"]),
                    volume     = int(r["tick_volume"]),
                )
                for r in rates
            ]
        except Exception as exc:
            raise exc

    def get_latest_bars(self, symbol: str, timeframe: str = "M1", count: int = 1) -> List[Bar]:
        self._ensure_connection()
        try:
            rates = mt5.copy_rates_from_pos(symbol, self._TF_MAP.get(timeframe, mt5.TIMEFRAME_M1), 0, count)
            if rates is None or len(rates) == 0:
                return []
            return [
                Bar(
                    timestamp  = datetime.fromtimestamp(int(r["time"])),
                    open_price = float(r["open"]),
                    high       = float(r["high"]),
                    low        = float(r["low"]),
                    close      = float(r["close"]),
                    volume     = int(r["tick_volume"]),
                )
                for r in rates
            ]
        except Exception:
            return []


# ---------------------------------------------------------------------------
# FVGDetector
# ---------------------------------------------------------------------------

class FVGDetector:
    """
    Detects Fair Value Gaps (FVGs) across symbols and timeframes.

    Bullish FVG : low(bar[i]) > high(bar[i-2])
    Bearish FVG : high(bar[i]) < low(bar[i-2])

    Cache: <paths.fvgs>/<SYMBOL>_fvgs.json
    Layout: self.fvgs[symbol][timeframe] = [fvg_dict, ...]
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
    ):
        self.symbols    = symbols    or settings.symbols
        self.timeframes = timeframes or [settings.fvg_timeframe]
        self.fetcher    = DataFetcher()

        fvg_cfg              = settings.orderblock_fvg_pip_size
        self.min_gap_pips: float = fvg_cfg.get("min", 3.5)
        self.max_gap_pips: float = fvg_cfg.get("max", 5.0)
        self.pip_size: float     = 0.0001

        self.fvg_dir = Path(settings.path_fvgs)
        self.fvgs: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self.load_fvgs_from_cache()

    def _generate_fvg_id(self, fvg: Dict[str, Any]) -> str:
        return (
            f"{fvg.get('type', 'unknown')}_{fvg.get('bar_open_time', '')}"
            f"_{fvg.get('high', 0):.5f}_{fvg.get('low', 0):.5f}"
            f"_{fvg.get('size_pips', 0):.2f}"
        )

    def _fvg_exists(self, new_fvg: Dict[str, Any], existing: List[Dict[str, Any]]) -> bool:
        new_id = self._generate_fvg_id(new_fvg)
        return any(self._generate_fvg_id(f) == new_id for f in existing)

    def _merge_fvg_list(
        self,
        existing: List[Dict[str, Any]],
        new_fvgs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        for fvg in new_fvgs:
            if not self._fvg_exists(fvg, existing):
                existing.append(fvg)
        return existing

    def _ensure_symbol_storage(self, symbol: str) -> None:
        if symbol not in self.fvgs:
            self.fvgs[symbol] = {tf: [] for tf in self.timeframes}

    def detect_fvgs_in_bars(self, bars: List[Bar]) -> List[Dict[str, Any]]:
        fvg_list: List[Dict[str, Any]] = []
        for i in range(2, len(bars)):
            bar1, bar3 = bars[i - 2], bars[i]
            if bar3.low > bar1.high:
                gap_pips = self.get_pip_diff(bar3.low, bar1.high)
                if self.min_gap_pips <= gap_pips <= self.max_gap_pips:
                    fvg_list.append({
                        "type": "bullish", "high": bar3.low, "low": bar1.high,
                        "size_pips": gap_pips,
                        "bar_open_time":    bar1.timestamp.isoformat(),
                        "detection_time":   bar3.timestamp.isoformat(),
                        "filled_timestamp": None,
                    })
            if bar3.high < bar1.low:
                gap_pips = self.get_pip_diff(bar3.high, bar1.low)
                if self.min_gap_pips <= gap_pips <= self.max_gap_pips:
                    fvg_list.append({
                        "type": "bearish", "high": bar1.low, "low": bar3.high,
                        "size_pips": gap_pips,
                        "bar_open_time":    bar1.timestamp.isoformat(),
                        "detection_time":   bar3.timestamp.isoformat(),
                        "filled_timestamp": None,
                    })
        return fvg_list

    def detect(
        self,
        start_dt: datetime,
        end_dt: datetime,
        timeframes: Optional[List[str]] = None,
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        tf_list     = timeframes or self.timeframes
        symbol_list = symbols    or self.symbols
        results: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

        for symbol in symbol_list:
            symbol_results: Dict[str, List[Dict[str, Any]]] = {}
            for tf in tf_list:
                try:
                    bars = self.fetcher.fetch_bars_from_mt5(start_dt, end_dt, symbol, tf)
                    fvgs = self.detect_fvgs_in_bars(bars) if bars else []
                    symbol_results[tf] = fvgs
                    self._ensure_symbol_storage(symbol)
                    self.fvgs[symbol][tf] = self._merge_fvg_list(self.fvgs[symbol][tf], fvgs)
                except Exception:
                    symbol_results[tf] = []
            results[symbol] = symbol_results

        self.clean_filled_fvgs()
        self.save_fvgs_to_cache()
        return results

    def clean_filled_fvgs(self) -> None:
        for symbol, tf_results in self.fvgs.items():
            self._ensure_symbol_storage(symbol)
            for tf, current_fvgs in tf_results.items():
                if tf not in self.fvgs[symbol]:
                    self.fvgs[symbol][tf] = []
                try:
                    if self.fvgs[symbol][tf]:
                        oldest = min(
                            datetime.fromisoformat(f["bar_open_time"])
                            for f in self.fvgs[symbol][tf]
                        )
                        check_bars = self.fetcher.fetch_bars_from_mt5(oldest, datetime.now(), symbol, "M5")
                    else:
                        check_bars = []

                    active_fvgs: List[Dict[str, Any]] = []
                    for fvg in self.fvgs[symbol][tf]:
                        if fvg.get("filled_timestamp") is None:
                            is_filled, filled_time = self.is_fvg_filled(fvg, check_bars)
                            if is_filled:
                                fvg["filled_timestamp"] = filled_time
                        active_fvgs.append(fvg)

                    self.fvgs[symbol][tf] = self._merge_fvg_list(active_fvgs, current_fvgs)
                except Exception:
                    self.fvgs[symbol][tf] = self._merge_fvg_list(self.fvgs[symbol][tf], current_fvgs)

    def is_fvg_filled(self, fvg: Dict[str, Any], bars: List[Bar]):
        fvg_type       = fvg["type"]
        fvg_high       = fvg["high"]
        fvg_low        = fvg["low"]
        detection_time = datetime.fromisoformat(fvg["detection_time"]) + timedelta(minutes=25)

        for bar in bars:
            if bar.timestamp < detection_time:
                continue
            if fvg_type == "bullish" and bar.low  <= fvg_high:
                return True, bar.timestamp.isoformat()
            if fvg_type == "bearish" and bar.high >= fvg_low:
                return True, bar.timestamp.isoformat()
        return False, None

    def get_fvg_cache_path(self, symbol: str) -> Path:
        return self.fvg_dir / f"{symbol}_fvgs.json"

    def save_fvgs_to_cache(self) -> None:
        try:
            self.fvg_dir.mkdir(parents=True, exist_ok=True)
            for symbol, tf_dict in self.fvgs.items():
                serializable: Dict[str, List[Dict[str, Any]]] = {}
                for tf, fvg_list in tf_dict.items():
                    serializable[tf] = []
                    for fvg in fvg_list:
                        fvg_copy = fvg.copy()
                        for ts_key in ("bar_open_time", "detection_time"):
                            if not isinstance(fvg_copy.get(ts_key), str):
                                fvg_copy[ts_key] = fvg_copy[ts_key].isoformat()
                        serializable[tf].append(fvg_copy)
                with open(self.get_fvg_cache_path(symbol), "w", encoding="utf-8") as fh:
                    json.dump(serializable, fh, indent=2)
        except Exception:
            pass

    def load_fvgs_from_cache(self) -> None:
        self.fvgs = {}
        try:
            for symbol in self.symbols:
                cache_path = self.get_fvg_cache_path(symbol)
                if cache_path.exists():
                    with open(cache_path, "r", encoding="utf-8") as fh:
                        self.fvgs[symbol] = json.load(fh)
                else:
                    self.fvgs[symbol] = {tf: [] for tf in self.timeframes}
        except Exception:
            for symbol in self.symbols:
                self.fvgs[symbol] = {tf: [] for tf in self.timeframes}

    def get_active_fvgs(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        symbols = [symbol] if symbol else list(self.fvgs.keys())
        result: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for sym in symbols:
            if sym not in self.fvgs:
                continue
            if timeframe:
                active = [f for f in self.fvgs[sym].get(timeframe, []) if f.get("filled_timestamp") is None]
                if active:
                    result.setdefault(sym, {})[timeframe] = active
            else:
                for tf, fvg_list in self.fvgs[sym].items():
                    active = [f for f in fvg_list if f.get("filled_timestamp") is None]
                    if active:
                        result.setdefault(sym, {})[tf] = active
        return result

    def get_pip_diff(self, price1: float, price2: float) -> float:
        return abs(price1 - price2) / self.pip_size


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(version=settings.system_version, prog_name=settings.system_name)
@click.option("--config",  "-c", type=click.Path(exists=True), help="Path to YAML configuration file")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose (DEBUG) output")
@click.option("--quiet",   "-q", is_flag=True, help="Suppress all output except warnings/errors")
@click.pass_context
def two_hunters_cli(ctx, config, verbose, quiet):
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"]   = quiet

    if config:
        settings.load_config(config)
    if verbose:
        settings.set("logging.level", "DEBUG")
    elif quiet:
        settings.set("logging.level", "WARNING")

    # Re-initialise logging now that verbose/quiet flags are known.
    _build_logger(verbose=verbose, quiet=quiet)

    log = get_logger()
    log.info("─" * 60)
    log.info("%s  v%s  starting up", settings.system_name, settings.system_version)
    if verbose:
        log.info("Mode: VERBOSE  (all debug messages visible)")
    elif quiet:
        log.info("Mode: QUIET    (warnings and errors only on console)")


@two_hunters_cli.command()
@click.option("--symbol",         "-s",  multiple=True, help="Trading symbol (e.g. EURUSD.). Defaults to config symbols.")
@click.option("--risk",           "-r",  type=float,    help="Risk percent per trade (e.g. 0.01). Defaults to config risk_percent.")
@click.option("--max-concurrent", "-mc", type=int, default=5,  help="Maximum concurrent symbol threads (default: 5).")
@click.option("--check-interval", "-ci", type=int, default=5,  help="Main-loop check interval in seconds (default: 5).")
@click.pass_context
def live(ctx, symbol, risk, max_concurrent, check_interval):
    log     = get_logger()
    fetcher = DataFetcher()
    symbols = list(symbol) if symbol else settings.symbols
    risk    = risk or settings.risk_percent
    verbose = ctx.obj.get("verbose", False)
    quiet   = ctx.obj.get("quiet",   False)

    if not (0 < risk <= 0.1):
        log.error("risk must be between 0 and 0.1, got %s", risk)
        sys.exit(1)

    mt5_conn = MT5Connection()
    if not mt5_conn.initialize_connection():
        log.error("Failed to initialize MT5 connection")
        sys.exit(1)
    log.info("MT5 connection established")

    account_info = mt5_conn.get_account_info()
    if account_info and hasattr(account_info, "balance"):
        balance = float(account_info.balance)
        log.info(
            "Account: %s  |  Balance: $%s  |  Leverage: 1:%s",
            getattr(account_info, "login",    "?"),
            "{:.2f}".format(balance),
            getattr(account_info, "leverage", "?"),
        )
    else:
        log.error("Unable to retrieve account balance from MT5")
        sys.exit(1)

    symbol_str = "  ".join(s.rstrip(".").upper() for s in symbols)
    log.info("Mode: LIVE  |  Symbols: %s  |  Risk: %.2f%%", symbol_str, risk * 100)
    log.debug("Max threads: %d  |  Check interval: %ds", max_concurrent, check_interval)

    try:
        session_start_t = datetime.strptime(settings.session_main_start, "%H:%M").time()
        session_end_t   = datetime.strptime(settings.session_main_end,   "%H:%M").time()
    except (ValueError, TypeError) as exc:
        log.error("Invalid session times in config: %s", exc)
        sys.exit(1)

    budget      = Budget(initial_balance=balance, initial_risk_percent=risk)
    stop_event  = threading.Event()
    thread_lock = threading.Lock()

    # ── Per-symbol thread registry ────────────────────────────────────
    # Key: symbol  →  {"thread": Thread, "date": date}
    thread_registry: Dict[str, Dict] = {}

    _last_wait_log: Optional[datetime] = None

    log.info("Admin commands active — edit admin_commands in YAML to control the system")
    log.info("Press Ctrl+C to stop")
    log.info("─" * 60)

    try:
        while not stop_event.is_set():

            # ── Admin commands — checked at the very top of every iteration ──
            admin.poll_and_apply(
                stop_event      = stop_event,
                thread_registry = thread_registry,
                mt5_conn        = mt5_conn,
            )

            if admin._halt_requested:
                time.sleep(check_interval)
                continue

            # ── Handle reset: stop all threads, clear registry, restart ──
            if admin.reset_requested:
                admin.clear_reset()
                log.warning("Executing reset — stopping all symbol threads")
                stop_event.set()
                for sym, reg in thread_registry.items():
                    t = reg["thread"]
                    if t.is_alive():
                        t.join(timeout=10)
                        log.info("  %s thread stopped for reset", sym)
                thread_registry.clear()
                stop_event.clear()
                log.warning("Reset complete — restarting with fresh context")
                try:
                    session_start_t = datetime.strptime(settings.session_main_start, "%H:%M").time()
                    session_end_t   = datetime.strptime(settings.session_main_end,   "%H:%M").time()
                except Exception:
                    pass
                continue

            if stop_event.is_set():
                break

            # ── Fetch broker time ──────────────────────────────────────
            latest = fetcher.get_latest_bars(symbols[0]) if symbols else []
            if not latest:
                time.sleep(check_interval)
                continue

            current_dt   = latest[-1].timestamp
            current_date = current_dt.date()
            weekday      = current_dt.weekday()   # 0=Mon … 4=Fri, 5=Sat, 6=Sun

            # ── Weekend guard ──────────────────────────────────────────
            if weekday >= 5:
                days_until_monday = 7 - weekday
                monday_open = datetime.combine(
                    current_date + timedelta(days=days_until_monday),
                    dt_time(0, 5),
                )
                wait_secs = (monday_open - current_dt).total_seconds()
                if wait_secs < 0:
                    wait_secs = 3600
                log.info(
                    "Weekend — markets closed. Sleeping %.1f hours until Monday open.",
                    wait_secs / 3600,
                )
                time.sleep(wait_secs - 1800 if wait_secs > 1800 else check_interval)
                continue

            # ── Check trading window ───────────────────────────────────
            current_time_only = current_dt.time()
            in_trading_window = session_start_t <= current_time_only <= session_end_t

            if in_trading_window:
                with thread_lock:
                    for sym in symbols:
                        reg = thread_registry.get(sym)

                        already_ran_today = (
                            reg is not None
                            and reg["date"] == current_date
                        )

                        if already_ran_today:
                            continue

                        # Clean up any leftover thread from a previous day
                        if reg and not reg["thread"].is_alive():
                            reg["thread"].join(timeout=0)

                        try:
                            context = get_trading_context(sym, current_dt)
                            if context['mbox']["extrema_flag"]:
                                log.warning("MBox high/low is formed after %s for %s - Skiping trading for today (Use force_order to trade today).", settings.time_flag_hour, sym)
                            t = threading.Thread(
                                target = _run_symbol_live_trading,
                                args   = (sym, fetcher, budget, mt5_conn, context,
                                          stop_event, verbose, quiet),
                                name   = f"TwoHunters-{sym}-{current_date}",
                                daemon = False,
                            )
                            thread_registry[sym] = {"thread": t, "date": current_date}
                            t.start()
                            log.info("Started trading thread for %s [%s]", sym, current_date)
                        except Exception as exc:
                            log.error("Failed to start thread for %s: %s", sym, exc)

                time.sleep(check_interval)

            else:
                # ── Outside trading window ─────────────────────────────
                now = current_dt
                session_start_dt = datetime.combine(current_date, session_start_t)
                if session_start_dt <= current_dt:
                    session_start_dt += timedelta(days=1)
                wait_seconds = max((session_start_dt - current_dt).total_seconds(), 0)

                if _last_wait_log is None or (now - _last_wait_log).total_seconds() >= 3600:
                    _last_wait_log = now
                    hours   = int(wait_seconds // 3600)
                    minutes = int((wait_seconds % 3600) // 60)
                    log.info("Outside trading window — next open in %dh %dm", hours, minutes)

                    recent_orders = mt5_conn.get_today_signals()
                    if recent_orders:
                        sig = sorted(recent_orders, key=lambda s: s.timestamp, reverse=True)[0]
                        outcome = sig.outcome.value if sig.outcome else "PENDING"
                        log.info(
                            "Last order: %s %s @ %.5f  lot:%.2f  gain:$%.2f  (%s)",
                            sig.symbol, sig.action.value, sig.entry_price,
                            sig.entry_lot or 0, sig.gain or 0, outcome,
                        )

                sleep_time = wait_seconds - 300 if wait_seconds > 300 else check_interval
                time.sleep(check_interval)

    except KeyboardInterrupt:
        log.warning("KeyboardInterrupt — shutting down live trading")
        stop_event.set()

    except Exception as exc:
        import traceback
        tb = traceback.extract_tb(sys.exc_info()[2])[-1]
        log.error("Unhandled exception: %s  (file %s line %d)", exc, tb.filename, tb.lineno)
        if verbose:
            traceback.print_exc()
        sys.exit(1)

    finally:
        log.info("Waiting for symbol threads to finish ...")
        for sym, reg in thread_registry.items():
            t = reg["thread"]
            if t.is_alive():
                t.join(timeout=10)
                log.info("  %s thread closed", sym)
        mt5_conn.shutdown_connection()
        log.info("Live trading stopped gracefully")


# ---------------------------------------------------------------------------
# Trading context helpers
# ---------------------------------------------------------------------------

def get_trading_context(
    symbol: str,
    current_datetime: datetime,
) -> Dict[str, Any]:
    """
    Build a snapshot of the trading context for one symbol:
        - Previous business day London & New York session high/low
        - Today's MBox result (high, low, extrema time-flag)
        - Active FVGs within the lookup window
    """
    log     = get_logger(symbol)
    fetcher = DataFetcher()

    london_cfg  = settings.session_london
    newyork_cfg = settings.session_newyork

    london_start  = datetime.strptime(london_cfg["start"],  "%H:%M").time()
    london_end    = datetime.strptime(london_cfg["end"],    "%H:%M").time()
    newyork_start = datetime.strptime(newyork_cfg["start"], "%H:%M").time()
    newyork_end   = datetime.strptime(newyork_cfg["end"],   "%H:%M").time()

    weekday = current_datetime.weekday()
    if weekday == 0:
        previous_day = current_datetime - timedelta(days=3)
    elif weekday == 6:
        previous_day = current_datetime - timedelta(days=2)
    else:
        previous_day = current_datetime - timedelta(days=1)

    prev_day_start = datetime.combine(previous_day.date(),     dt_time(1, 0))
    prev_day_end   = datetime.combine(current_datetime.date(), dt_time(3, 30))

    london_result  = {"high": None, "low": None}
    newyork_result = {"high": None, "low": None}

    try:
        prev_bars = fetcher.fetch_bars_from_mt5(prev_day_start, prev_day_end, symbol)
        if prev_bars:
            london_start_dt  = datetime.combine(previous_day.date(), london_start)
            london_end_dt    = datetime.combine(previous_day.date(), london_end)
            newyork_start_dt = datetime.combine(previous_day.date(), newyork_start)
            newyork_end_dt   = datetime.combine(previous_day.date(), newyork_end)
            if newyork_end <= newyork_start:
                newyork_end_dt += timedelta(days=1)

            london_bars  = [b for b in prev_bars if london_start_dt  <= b.timestamp <= london_end_dt]
            newyork_bars = [b for b in prev_bars if newyork_start_dt <= b.timestamp <= newyork_end_dt]

            if london_bars:
                london_result  = {"high": max(b.high for b in london_bars),
                                  "low":  min(b.low  for b in london_bars)}
                log.debug("London prev-day H/L: %.5f / %.5f", london_result["high"], london_result["low"])
            if newyork_bars:
                newyork_result = {"high": max(b.high for b in newyork_bars),
                                  "low":  min(b.low  for b in newyork_bars)}
                log.debug("NY prev-day H/L: %.5f / %.5f", newyork_result["high"], newyork_result["low"])
    except Exception as exc:
        log.warning("Could not fetch previous-day bars: %s", exc)

    # Today's MBox
    mbox_start_dt = datetime.combine(
        current_datetime.date(),
        datetime.strptime(settings.mbox_start, "%H:%M").time(),
    )
    mbox_end_dt = datetime.combine(
        current_datetime.date(),
        datetime.strptime(settings.mbox_end, "%H:%M").time(),
    )
    mbox_result = {"min_val": None, "max_val": None, "extrema_flag": False}

    try:
        mbox_bars = fetcher.fetch_bars_from_mt5(mbox_start_dt, mbox_end_dt, symbol)
        if mbox_bars and len(mbox_bars) >= 20:
            all_ohlc = [(b.open, b.high, b.low, b.close) for b in mbox_bars]
            min_val  = min(min(ohlc) for ohlc in all_ohlc)
            max_val  = max(max(ohlc) for ohlc in all_ohlc)
            idx_min  = next(i for i, ohlc in enumerate(all_ohlc) if min_val in ohlc)
            idx_max  = next(i for i, ohlc in enumerate(all_ohlc) if max_val in ohlc)
            ts_min   = mbox_bars[idx_min].timestamp
            ts_max   = mbox_bars[idx_max].timestamp

            tf_hour      = datetime.strptime(settings.time_flag_hour, "%H:%M").time()
            extrema_flag = ts_max.time() >= tf_hour or ts_min.time() >= tf_hour

            mbox_result = {"max_val": max_val, "min_val": min_val, "extrema_flag": extrema_flag}
            log.debug(
                "MBox H/L: %.5f / %.5f  extrema_flag=%s  (bars=%d)",
                max_val, min_val, extrema_flag, len(mbox_bars),
            )
        else:
            log.debug("MBox: insufficient bars (%d)", len(mbox_bars) if mbox_bars else 0)
    except Exception as exc:
        log.warning("Could not build MBox: %s", exc)

    # Active FVGs
    fvg_start    = current_datetime - timedelta(days=settings.lookup_days)
    fvg_detector = FVGDetector(symbols=[symbol], timeframes=[settings.fvg_timeframe])
    active_fvgs: List[Dict[str, Any]] = []

    try:
        fvg_detector.detect(
            start_dt   = fvg_start,
            end_dt     = current_datetime,
            timeframes = [settings.fvg_timeframe],
            symbols    = [symbol],
        )

        for fvg in fvg_detector.fvgs.get(symbol, {}).get(settings.fvg_timeframe, []):
            filled_ts = fvg.get("filled_timestamp")
            is_active = filled_ts is None
            if not is_active:
                try:
                    is_active = datetime.fromisoformat(filled_ts) > current_datetime
                except (ValueError, TypeError):
                    is_active = False
            if not is_active:
                continue

            det_time_str = fvg.get("detection_time")
            if det_time_str:
                try:
                    det_time = datetime.fromisoformat(det_time_str)
                    if not (fvg_start <= det_time <= current_datetime):
                        continue
                except (ValueError, TypeError):
                    continue

            active_fvgs.append({
                "type":             fvg.get("type"),
                "high":             fvg.get("high"),
                "low":              fvg.get("low"),
                "size_pips":        fvg.get("size_pips"),
                "bar_open_time":    fvg.get("bar_open_time"),
                "detection_time":   det_time_str,
                "filled_timestamp": filled_ts,
            })

        log.info("Context ready — MBox OK  London OK  NY OK  FVGs active: %d", len(active_fvgs))
    except Exception as exc:
        log.warning("Could not build FVG context: %s", exc)

    return {
        "london":      london_result,
        "newyork":     newyork_result,
        "mbox":        mbox_result,
        "active_fvgs": active_fvgs,
    }


# ---------------------------------------------------------------------------
# Per-symbol live trading thread
# ---------------------------------------------------------------------------

def _run_symbol_live_trading(
    symbol: str,
    fetcher: DataFetcher,
    budget: Budget,
    mt5_conn: MT5Connection,
    results: dict,
    stop_event: threading.Event,
    verbose: bool = False,
    quiet: bool   = False,
):
    log = get_logger(symbol)

    mbox_result     = results.get("mbox",        {})
    newyork_results = results.get("newyork",     {})
    london_results  = results.get("london",      {})
    fvg_results     = results.get("active_fvgs", [])

    _num_hunt_main : int  = settings.num_hunt_main
    _should_recover: bool = settings.should_recover

    # ------------------------------------------------------------------
    # Nested helpers
    # ------------------------------------------------------------------

    def _is_order_bar(prev_bar: Bar, this_bar: Bar, direction: str) -> Optional[Bar]:
        if this_bar.is_weak:
            return None
        _m = settings.orderblock_fvg_pip_size.get("min", 3.5) / 10000
        if direction == "SELL":
            if this_bar.close < prev_bar.low - (prev_bar.range * _m) and this_bar.is_bearish:
                return this_bar
        elif direction == "BUY":
            if this_bar.close > prev_bar.high + (prev_bar.range * _m) and this_bar.is_bullish:
                return this_bar
        return None

    def find_signal_bar(bars: List[Bar], hunter_bar: Bar, direction: str) -> Optional[Bar]:
        hunter_idx = next(
            (i for i, b in enumerate(bars) if b.timestamp == hunter_bar.timestamp), None
        )
        if hunter_idx is None:
            return None
        for i in range(hunter_idx + 1, len(bars)):
            search_idx = i - 1
            while search_idx >= 0 and bars[search_idx].is_weak:
                search_idx -= 1
            if search_idx < 0:
                continue
            result = _is_order_bar(bars[search_idx], bars[i], direction)
            if result:
                return result
        return None

    def find_extrema_bar(hunter_bar: Bar, signal_bar: Bar, bars: List[Bar], action: str) -> float:
        extrema_bars = [b for b in bars if hunter_bar.timestamp <= b.timestamp <= signal_bar.timestamp]
        if not extrema_bars:
            return hunter_bar.high if action == "SELL" else hunter_bar.low
        if action == "SELL":
            return max(extrema_bars, key=lambda b: b.high).high
        return min(extrema_bars, key=lambda b: b.low).low

    def default_breakout(
        session_bars: List[Bar],
        box: dict,
        num_hunt: int = 2,
    ) -> Tuple[Optional[float], Optional[Bar], Optional[str], Optional[Bar], Optional[float]]:
        if not session_bars or not box:
            return None, None, None, None, None

        min_val = box.get("min_val")
        max_val = box.get("max_val")
        if min_val is None or max_val is None:
            return None, None, None, None, None

        lookahead           = 1
        breakout_stage_up   = 0
        breakout_stage_down = 0
        _15_triggered       = False

        start    = session_bars[0].timestamp
        end      = session_bars[-1].timestamp
        _15_bars = fetcher.fetch_bars_from_mt5(start, end, symbol, "M15")

        _skip_minutes             = settings.skip_minutes
        _15_first_hunt_ts         = None
        _skip_threshold_triggered = False

        m = len(_15_bars)
        for i, bar in enumerate(_15_bars):
            if i + lookahead >= m:
                break
            skip_threshold = (
                datetime.combine(bar.timestamp.date(), start.time()) +
                timedelta(minutes=_skip_minutes)
            ).time()
            if bar.timestamp.time() < skip_threshold:
                continue
            is_local_max = all(_15_bars[j].high < bar.high for j in range(i + 1, min(i + 1 + lookahead, m)))
            is_local_min = all(_15_bars[j].low  > bar.low  for j in range(i + 1, min(i + 1 + lookahead, m)))
            if is_local_max and bar.high > max_val:
                _15_first_hunt_ts = bar.timestamp
                break
            if is_local_min and bar.low < min_val:
                _15_first_hunt_ts = bar.timestamp
                break

        def _m15_confirmed(bar: Bar) -> bool:
            return _15_first_hunt_ts is not None and bar.timestamp > _15_first_hunt_ts

        n = len(session_bars)
        for i, bar in enumerate(session_bars):
            if i + lookahead >= n:
                break

            is_local_max = all(bar.high > session_bars[j].high for j in range(i + 1, min(i + 1 + lookahead, n)))
            is_local_min = all(bar.low  < session_bars[j].low  for j in range(i + 1, min(i + 1 + lookahead, n)))

            skip_threshold = (
                datetime.combine(bar.timestamp.date(), start.time()) +
                timedelta(minutes=_skip_minutes)
            ).time()

            if is_local_max and bar.high > max_val:
                breakout_stage_up += 1
                max_val = bar.high
                log.debug("Breakout UP stage %d/%d @ %.5f", breakout_stage_up, num_hunt, bar.high)
                if not _skip_threshold_triggered and bar.timestamp.time() < skip_threshold:
                    _skip_threshold_triggered = True
                    num_hunt += 1
                if breakout_stage_up >= num_hunt:
                    signal_bar = find_signal_bar(session_bars, bar, "SELL")
                    if signal_bar:
                        if not _15_triggered and not _m15_confirmed(bar):
                            num_hunt     += 1
                            _15_triggered = True
                            continue
                        extrema = max(max_val, find_extrema_bar(bar, signal_bar, session_bars, "SELL"))
                        return extrema, signal_bar, "SELL", bar, None

            elif is_local_min and bar.low < min_val:
                breakout_stage_down += 1
                min_val = bar.low
                log.debug("Breakout DOWN stage %d/%d @ %.5f", breakout_stage_down, num_hunt, bar.low)
                if not _skip_threshold_triggered and bar.timestamp.time() < skip_threshold:
                    _skip_threshold_triggered = True
                    num_hunt += 1
                if breakout_stage_down >= num_hunt:
                    signal_bar = find_signal_bar(session_bars, bar, "BUY")
                    if signal_bar:
                        if not _15_triggered and not _m15_confirmed(bar):
                            num_hunt     += 1
                            _15_triggered = True
                            continue
                        extrema = min(min_val, find_extrema_bar(bar, signal_bar, session_bars, "BUY"))
                        return extrema, signal_bar, "BUY", bar, None

        return None, None, None, None, None

    def fvg_breakout(
        session_bars: List[Bar],
        mbox_result: dict,
        london_results: dict,
        newyork_results: dict,
        fvg_results: list,
    ) -> Tuple[Optional[float], Optional[Bar], Optional[str], Optional[Bar], Optional[float]]:
        if not session_bars or not mbox_result:
            return None, None, None, None, None

        mbox_min = mbox_result.get("min_val")
        mbox_max = mbox_result.get("max_val")
        if mbox_min is None or mbox_max is None:
            return None, None, None, None, None

        all_fvgs        = fvg_results or []
        first_bar       = session_bars[0]
        mid             = (mbox_min + mbox_max) / 2
        is_bullish_side = first_bar.close > mid
        lookahead       = 1

        if is_bullish_side:
            candidates = [((f["high"] + f["low"]) / 2, "bullish_fvg_mid")
                          for f in all_fvgs if f["type"] == "bullish"]
            if london_results.get("low")  is not None: candidates.append((london_results["low"],   "london_low"))
            if newyork_results.get("low") is not None: candidates.append((newyork_results["low"],  "newyork_low"))
            candidates.append((mbox_min, "mbox_min"))
            below = [(p, s) for p, s in candidates if p < first_bar.low]
            if not below:
                return None, None, None, None, None
            breakout_line = min(below, key=lambda x: abs(x[0] - first_bar.low))
        else:
            candidates = [((f["high"] + f["low"]) / 2, "bearish_fvg_mid")
                          for f in all_fvgs if f["type"] == "bearish"]
            if london_results.get("high")  is not None: candidates.append((london_results["high"],  "london_high"))
            if newyork_results.get("high") is not None: candidates.append((newyork_results["high"], "newyork_high"))
            candidates.append((mbox_max, "mbox_max"))
            above = [(p, s) for p, s in candidates if p > first_bar.high]
            if not above:
                return None, None, None, None, None
            breakout_line = max(above, key=lambda x: abs(first_bar.high - x[0]))

        breakout_bar = breakout_index = None
        for i, bar in enumerate(session_bars):
            if i < lookahead:
                continue
            if is_bullish_side and bar.low < breakout_line[0]:
                breakout_bar, breakout_index = bar, i
                break
            elif not is_bullish_side and bar.high > breakout_line[0]:
                breakout_bar, breakout_index = bar, i
                break
        if breakout_bar is None:
            return None, None, None, None, None

        extrema_bar = extrema_price = extrema_index = None
        for i in range(breakout_index, len(session_bars) - 1):
            bar      = session_bars[i]
            prev_bar = session_bars[i - 1]
            next_bar = session_bars[i + 1]
            if is_bullish_side:
                if bar.low < prev_bar.low - 0.00001 and bar.low < next_bar.low - 0.00001:
                    extrema_bar, extrema_price, extrema_index = bar, bar.low, i
                    break
            else:
                if bar.high > prev_bar.high + 0.00001 and bar.high > next_bar.high + 0.00001:
                    extrema_bar, extrema_price, extrema_index = bar, bar.high, i
                    break
        if extrema_bar is None:
            return None, None, None, None, None

        signal_bar = None
        for i in range(extrema_index + 1, len(session_bars)):
            signal_bar = _is_order_bar(session_bars[i - 1], session_bars[i], "BUY" if is_bullish_side else "SELL")
            if signal_bar:
                break
        if signal_bar is None:
            return None, None, None, None, None

        return extrema_price, signal_bar, "BUY" if is_bullish_side else "SELL", breakout_bar, None

    def calculate_entry_details(
        action: SignalAction,
        signal_bar: Bar,
        extrema: float,
    ) -> Tuple[float, float, float]:
        margin   = settings.margin_pips * budget.pip_size
        sl_ratio = settings.sl_ratio
        tp_ratio = settings.tp_ratio
        _r = abs(extrema - signal_bar.close)

        if action == SignalAction.SELL:
            entry_price = signal_bar.close
            stop_loss   = entry_price + _r * sl_ratio + margin
            take_profit = entry_price - _r * tp_ratio
        else:
            entry_price = signal_bar.close
            stop_loss   = extrema - _r * sl_ratio - margin
            take_profit = entry_price + _r * tp_ratio

        return entry_price, stop_loss, take_profit

    def create_signal(
        action: SignalAction,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        timestamp: datetime,
        signal_type: SignalType,
    ) -> Optional[Signal]:
        if action == SignalAction.SELL and not (take_profit < entry_price < stop_loss):
            return None
        if action == SignalAction.BUY  and not (stop_loss < entry_price < take_profit):
            return None
        return Signal(
            action=action, entry_price=entry_price, stop_loss=stop_loss,
            take_profit=take_profit, symbol=symbol, timestamp=timestamp,
            signal_type=signal_type,
        )

    def attempt_signal(
        failed_signal: Optional[Signal] = None,
        session_bars: Optional[List[Bar]] = None,
    ) -> Optional[Signal]:
        if not session_bars:
            return None

        if failed_signal is None:
            extrema, signal_bar, action, hunter_bar, _ = default_breakout(
                session_bars, mbox_result, num_hunt=_num_hunt_main
            )
        else:
            bars_after = [b for b in session_bars if b.timestamp > failed_signal.outcome_timestamp]
            extrema, signal_bar, action, hunter_bar, _ = fvg_breakout(
                session_bars    = bars_after,
                mbox_result     = mbox_result,
                london_results  = london_results,
                newyork_results = newyork_results,
                fvg_results     = fvg_results,
            )

        if not signal_bar:
            return None

        action_enum = SignalAction.SELL if action == "SELL" else SignalAction.BUY
        entry_price, stop_loss, take_profit = calculate_entry_details(action_enum, signal_bar, extrema)

        sig = create_signal(
            action_enum, entry_price, stop_loss, take_profit,
            signal_bar.timestamp,
            SignalType.MAIN if failed_signal is None else SignalType.RECOVERY,
        )
        if not sig:
            return None

        if failed_signal is None:
            sig.time_flag = mbox_result.get("extrema_flag", False)
            sig.trend     = mbox_result.get("trend", False)
        else:
            sig.trend      = failed_signal.trend
            sig.fake_CHoCH = failed_signal.fake_CHoCH
            sig.time_flag  = failed_signal.time_flag

        budget.update_risk_percent(sig)
        sig.stop_loss_pips   = budget.pips_from_diff(abs(sig.entry_price - sig.stop_loss))
        sig.take_profit_pips = budget.pips_from_diff(abs(sig.take_profit - sig.entry_price))
        sig.entry_lot        = budget.lots_from_diff(sig.symbol, abs(sig.entry_price - sig.stop_loss))
        return sig

    def _fetch_and_update_bars(last_bar_time=None, bars=None) -> List[Bar]:
        if bars is None:
            bars = []
        try:
            if not bars:
                current_bars = fetcher.get_latest_bars(symbol=symbol)
                if not current_bars:
                    return []
                broker_time   = current_bars[-1].timestamp
                mbox_start_dt = datetime.combine(
                    broker_time.date(),
                    datetime.strptime(settings.mbox_start, "%H:%M").time(),
                )
                mbox_bars = fetcher.fetch_bars_from_mt5(
                    start_dt  = mbox_start_dt,
                    end_dt    = broker_time,
                    symbol    = symbol,
                    timeframe = settings.timeframe,
                )
                if mbox_bars:
                    bars.extend(mbox_bars)
                return bars

            fetched = fetcher.get_latest_bars(symbol=symbol, count=10)
            if not fetched:
                return bars
            current_time = fetched[-1].timestamp
            for bar in fetched:
                if last_bar_time and bar.timestamp <= last_bar_time:
                    continue
                if (current_time - bar.timestamp).total_seconds() < 60:
                    continue
                bars.append(bar)
            return bars
        except Exception:
            return bars


    # ------------------------------------------------------------------
    # Main per-symbol loop
    # ------------------------------------------------------------------

    signals_dir = Path(settings.path_signals)
    signals_dir.mkdir(parents=True, exist_ok=True)

    last_signal_date: Optional[datetime.date] = None
    signal: Optional[Signal]                  = None
    check_interval: int                       = settings.work_interval

    budget.calculate_pip_size(symbol)
    budget.calculate_lot_size(symbol)
    bars: List[Bar] = []

    # Reload today's already-placed signal to avoid duplicate orders after restart
    try:
        _latest: List[Bar] = []
        while not _latest:
            _latest = fetcher.get_latest_bars(symbol)
            if not _latest:
                time.sleep(1)
        current_date = _latest[-1].timestamp.date()

        for signal_file in signals_dir.glob(f"{symbol}_*.json"):
            try:
                parts = signal_file.stem.split("_")
                if len(parts) >= 2:
                    file_date = datetime.strptime(parts[1], "%Y%m%d").date()
                    if file_date == current_date:
                        with open(signal_file, "r") as fh:
                            signal = Signal.from_dict(json.load(fh))
                        last_signal_date = current_date
                        log.info(
                            "Loaded existing signal: %s @ %.5f",
                            signal.action.value, signal.entry_price,
                        )
            except Exception as exc:
                log.warning("Error loading signal file %s: %s", signal_file.name, exc)
    except Exception as exc:
        log.warning("Error loading today's signals: %s", exc)

    _last_no_signal_log: Optional[datetime] = None
    _last_pause_log:     Optional[datetime] = None

    log.info("Thread started — scanning for signals")

    try:
        while not stop_event.is_set():
            # Fetch current broker time
            _latest = []
            while not _latest and not stop_event.is_set():
                _latest = fetcher.get_latest_bars(symbol)
                if not _latest:
                    time.sleep(1)

            if stop_event.is_set():
                log.info("Thread died — waiting for tommorow")
                break

            current_datetime = _latest[-1].timestamp
            current_date     = current_datetime.date()

            # ── Already traded today ───────────────────────────────────
            if last_signal_date == current_date:
                if signal:
                    recent_orders = mt5_conn.get_today_signals()
                    for sig in recent_orders:
                        if sig.ticket == signal.ticket:
                            if sig.outcome != signal.outcome:
                                log.info(
                                    "Signal outcome updated: %s → %s  gain=$%.2f",
                                    signal.outcome.value if signal.outcome else "?",
                                    sig.outcome.value,
                                    sig.gain or 0,
                                )
                            signal = sig
                            break
                log.info("Order already placed — waiting for tommorow")
                break

            bars = _fetch_and_update_bars(
                last_bar_time = bars[-1].timestamp if bars else None,
                bars          = bars,
            )

            new_signal = attempt_signal(session_bars=bars)
            if new_signal is None:
                now = datetime.now()
                if _last_no_signal_log is None or (now - _last_no_signal_log).total_seconds() >= 300:
                    _last_no_signal_log = now
                    log.debug("Scanning ... no signal yet  (bars collected: %d)", len(bars))
                time.sleep(check_interval)
                continue

            signal = new_signal
            log.info(
                "Signal found: %s @ %.5f  SL=%.5f  TP=%.5f  lot=%.2f  tf=%s",
                signal.action.value, signal.entry_price,
                signal.stop_loss, signal.take_profit,
                signal.entry_lot or 0,
                "YES" if signal.time_flag else "NO",
            )

            # ── Pause check ────────────────────────────────────────────
            if admin.is_paused(symbol):
                norm_sym = symbol.rstrip(".").upper()
                with admin._lock:
                    admin.pending_signals[norm_sym] = signal
                now = datetime.now()
                if _last_pause_log is None or (now - _last_pause_log).total_seconds() >= 300:
                    _last_pause_log = now
                    log.info("Symbol is PAUSED — signal stored, order placement suspended")
                time.sleep(check_interval)
                continue

            # ── Flag check ────────────────────────────────────────────
            if not admin.force_order and signal.time_flag:
                log.warning(f"Signal is flagged — MBox high/low was formed after {settings.time_flag_hour} — skipping order placement")
                time.sleep(check_interval)
                continue

            elif admin.force_order and signal.time_flag:
                log.warning(f"Signal is flagged — MBox high/low was formed after {settings.time_flag_hour} — but ADMIN is forcing order placement")

            # ── Place order with retry ─────────────────────────────────
            order_placed = False
            _timedelta   = timedelta(minutes=settings.retry_order_minutes + 1) if not admin.force_order else timedelta(days=5)
            deadline     = signal.timestamp + _timedelta
            attempt_num  = 0
            
            if not (current_datetime <= deadline):
                if admin.force_order:
                    log.warning(f"Order placement deadline has passed but placing order anyway — ADMIN")
                else:
                    log.warning(f"Order placement deadline passed by {round((current_datetime - deadline).total_seconds())}s — skipping order placement for today")
                    break
            
            while current_datetime <= deadline and not stop_event.is_set():
                attempt_num += 1
                log.debug("Order placement attempt %d ...", attempt_num)
                order_placed = mt5_conn.place_order(signal, force=admin.force_order)
                if order_placed:
                    log.info(
                        "Order placed: %s %s @ %.5f  ticket=%s",
                        signal.symbol, signal.action.value,
                        signal.entry_price, signal.ticket,
                    )
                    break
                time.sleep(1)

            if order_placed:
                last_signal_date = current_date
                signal_file      = signals_dir / f"{symbol}_{current_date.strftime('%Y%m%d')}.json"
                with open(signal_file, "w") as fh:
                    json.dump(signal.to_dict(), fh, indent=2)

                log.info("Order placed — waiting for tommorow")
                break

                if _should_recover and signal.outcome == SignalOutcome.LOSS:
                    bars_after = _fetch_and_update_bars(last_bar_time=signal.outcome_timestamp, bars=[])
                    recovery   = attempt_signal(failed_signal=signal, session_bars=bars_after)
                    if recovery and mt5_conn.place_order(recovery, force=admin.force_order):
                        log.info("Recovery order placed: %s @ %.5f", recovery.action.value, recovery.entry_price)

            else:
                log.warning(
                    "Failed to place order after %d minutes (%d attempts) — giving up for today",
                    settings.retry_order_minutes, attempt_num,
                )
                break

    except Exception as exc:
        log.error("Fatal error in trading loop: %s", exc, exc_info=True)

    log.info("Thread exiting and waiting for tommorow ...")

if __name__ == "__main__":
    two_hunters_cli()
