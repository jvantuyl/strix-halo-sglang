# systemd user units

Templates for running the server as a *user* unit (started by the user's
own systemd manager, kept alive across logouts and reboots with linger).
Nothing here is specific to one box: the deployment values go in an
environment file.

| File | Installs as | Purpose |
|---|---|---|
| `qwen38-sglang.service` | `~/.config/systemd/user/qwen38-sglang.service` | Runs `start-qwen38.sh` (the `docker run` in the foreground) with the variables from the env file |
| `qwen38-sglang.env.example` | `~/.config/qwen38-sglang.env` | Checkpoint, PLE cache dir, container name, request cap, extra sglang flags (MTP) |

## Install

```bash
# once, as root: keep the user's manager running without a login session
sudo loginctl enable-linger "$USER"

# the unit
install -d ~/.config/systemd/user
install -m 644 systemd/qwen38-sglang.service ~/.config/systemd/user/
install -m 600 systemd/qwen38-sglang.env.example ~/.config/qwen38-sglang.env
$EDITOR ~/.config/qwen38-sglang.env      # MODEL_DIR, PLE_DIR, names, cap, MTP flags
systemctl --user daemon-reload
systemctl --user enable --now qwen38-sglang.service
```

`qwen38-sglang.service` expects the repository at `~/strix-halo-sglang`;
edit `ExecStart` if it lives elsewhere. From a shell without a session
(`sudo -u`, cron) point `systemctl --user` at the manager with
`XDG_RUNTIME_DIR=/run/user/$(id -u)`.

## Notes

- A user unit cannot order itself after `docker.service` (a system unit),
  so `ExecStartPre` polls `docker info` until the daemon answers. Loading
  takes 3–4 minutes, or 2–3 with `QWEN38_PRESHARDED_DIR` set (one 7-minute
  boot per image to write the dump; see the runbook's Load time section);
  `Restart=on-failure` covers a failed load.
- Stopping the unit with requests in flight takes the full 120 s
  `TimeoutStopSec`: the server drains them until it is killed. An idle
  server stops in about 15 s.
- Stopping the unit sends SIGTERM to the `docker run` client, which proxies
  it to the server. `ExecStopPost` removes the container regardless, and
  the launcher removes a stale one of the same name on the next start.
- Logs: `journalctl --user -u qwen38-sglang -f` (the server log, including
  the load and the per-batch lines).
