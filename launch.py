"""Launch wrapper: pick models and config, then hand off to train.py.

Tries Textual for a form TUI. If not installed, falls back to printing
the equivalent CLI command for the user to copy/run manually.
"""
import os
import subprocess
import sys

TRAIN_PY = os.path.join(os.path.dirname(__file__), "train.py")


def fallback_cli():
    """Print a copy-pasteable command when Textual isn't available."""
    print("textual not installed — install with: pip install textual")
    print("Or run train.py directly. Example:\n")
    print(f"  python {TRAIN_PY} --models lstm cnn1d --epochs 15 --stride 15")
    print(f"\nAll flags: --window-size, --batch-size, --epochs, --lr,")
    print(f"  --patience, --stride, --num-workers, --models, --data-dir, --ckpt-dir")


def launch_tui():
    """Run the Textual form, build CLI args, and exec train.py."""
    from textual.app import App, ComposeResult
    from textual.widgets import (
        Checkbox, Input, Button, Static, Header, Footer, Label, Rule,
    )
    from textual.containers import Horizontal, Vertical, VerticalScroll

    class LaunchForm(App):
        CSS = """
        Screen { padding: 1 2; }
        Label { width: 100%; }
        .section { margin: 1 0; }
        .field-row { height: 3; margin: 0 0 1 0; }
        .field-row Label { width: 14; height: 3; content-align: left middle; }
        .field-row Input { width: 1fr; height: 3; }
        .model-row { height: 3; }
        .model-row Checkbox { width: auto; }
        .btn-row { height: 3; margin-top: 1; }
        .btn-row Button { width: 20; }
        .info { background: $surface; padding: 1 2; margin: 1 0; border: solid $primary; }
        """

        result_cmd = None

        def compose(self) -> ComposeResult:
            yield Header()
            with VerticalScroll():
                yield Static("Trading Arena — Training Launcher", classes="section")
                yield Rule()

                yield Static("Models to train:", classes="section")
                with Horizontal(classes="model-row"):
                    yield Checkbox("lstm", True, id="m-lstm")
                    yield Checkbox("cnn1d", True, id="m-cnn1d")
                    yield Checkbox("cnn_lstm", True, id="m-cnn_lstm")
                    yield Checkbox("transformer", True, id="m-transformer")
                yield Rule()

                yield Static("Hyperparameters:", classes="section")
                with Horizontal(classes="field-row"):
                    yield Label("window-size")
                    yield Input(value="60", id="window-size")
                with Horizontal(classes="field-row"):
                    yield Label("batch-size")
                    yield Input(value="256", id="batch-size")
                with Horizontal(classes="field-row"):
                    yield Label("epochs")
                    yield Input(value="15", id="epochs")
                with Horizontal(classes="field-row"):
                    yield Label("lr")
                    yield Input(value="1e-3", id="lr")
                with Horizontal(classes="field-row"):
                    yield Label("patience")
                    yield Input(value="4", id="patience")
                with Horizontal(classes="field-row"):
                    yield Label("stride")
                    yield Input(value="15", id="stride")
                with Horizontal(classes="field-row"):
                    yield Label("num-workers")
                    yield Input(value="0", id="num-workers")
                yield Rule()

                yield Static("Dataset info:", classes="section")
                yield Static(self._dataset_info(), id="info", classes="info")

                with Horizontal(classes="btn-row"):
                    yield Button("Start Training", variant="primary", id="start")
                    yield Button("Quit", variant="error", id="quit")
            yield Footer()

        def _dataset_info(self) -> str:
            lines = []
            try:
                from train import check_memory, load_meta, ensure_numpy_cache, LazyTickerWindows, compute_class_weights
                import os as _os
                data_dir = _os.environ.get("DATA_DIR", _os.path.join(_os.path.dirname(__file__), "data", "processed"))

                meta = load_meta(data_dir)
                feature_cols = meta["feature_cols"]
                n_feats = len(feature_cols)

                avail = check_memory()
                if avail is not None:
                    lines.append(f"Available RAM: {avail:.1f} GB")

                lines.append(f"Features: {n_feats}")
                lines.append(f"Feature cols: {', '.join(feature_cols[:8])}{'...' if n_feats > 8 else ''}")

                try:
                    train_cache = ensure_numpy_cache(data_dir, "train", feature_cols)
                    ds = LazyTickerWindows(train_cache, 60, 15)
                    lines.append(f"Train windows (stride=15, win=60): {len(ds):,}")
                except Exception as e:
                    lines.append(f"Train windows: (error: {e})")

                try:
                    cw = compute_class_weights(_os.path.join(data_dir, "train.parquet"))
                    lines.append(f"Class weights: {cw.numpy().round(3)}")
                except Exception:
                    pass

            except Exception as e:
                lines.append(f"(Could not load dataset info: {e})")

            return "\n".join(lines) if lines else "No dataset info available"

        def on_button_pressed(self, event):
            if event.button.id == "quit":
                self.exit()
                return

            if event.button.id == "start":
                models = []
                for mid in ["m-lstm", "m-cnn1d", "m-cnn_lstm", "m-transformer"]:
                    cb = self.query_one(f"#{mid}", Checkbox)
                    if cb.value:
                        models.append(cb.id.removeprefix("m-"))

                if not models:
                    self.notify("Select at least one model", severity="error")
                    return

                fields = {
                    "window-size": self.query_one("#window-size", Input).value,
                    "batch-size": self.query_one("#batch-size", Input).value,
                    "epochs": self.query_one("#epochs", Input).value,
                    "lr": self.query_one("#lr", Input).value,
                    "patience": self.query_one("#patience", Input).value,
                    "stride": self.query_one("#stride", Input).value,
                    "num-workers": self.query_one("#num-workers", Input).value,
                }

                cmd = [sys.executable, TRAIN_PY, "--models"] + models
                for k, v in fields.items():
                    flag = f"--{k}"
                    cmd.extend([flag, v])

                self.result_cmd = cmd
                self.exit()

    app = LaunchForm()
    app.run()

    if app.result_cmd:
        print(f"Running: {' '.join(app.result_cmd)}\n")
        subprocess.run(app.result_cmd)
    else:
        fallback_cli()


if __name__ == "__main__":
    try:
        import textual  # noqa: F401
        launch_tui()
    except ImportError:
        fallback_cli()
