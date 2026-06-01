"""Tests for leuk.cli.doctor — setup diagnostics."""

from __future__ import annotations

import leuk.cli.doctor as doc


class TestDistroPackaging:
    def test_distro_family_parses_os_release(self, monkeypatch):
        monkeypatch.setattr(doc, "_os_release", lambda: {"ID": "neon", "ID_LIKE": "ubuntu debian"})
        assert doc._distro_family() == {"neon", "ubuntu", "debian"}

    def test_pkg_install_apt_for_debian(self, monkeypatch):
        monkeypatch.setattr(doc, "_os_release", lambda: {"ID": "ubuntu", "ID_LIKE": "debian"})
        assert doc._pkg_install("ydotool") == ["sudo apt install -y ydotool"]

    def test_pkg_install_pacman_for_arch(self, monkeypatch):
        monkeypatch.setattr(doc, "_os_release", lambda: {"ID": "arch"})
        assert "pacman" in doc._pkg_install("ydotool")[0]

    def test_pkg_install_unknown_distro_generic(self, monkeypatch):
        monkeypatch.setattr(doc, "_os_release", lambda: {"ID": "weird"})
        assert "package manager" in doc._pkg_install("ydotool")[0]


class TestModernYdotoolInstall:
    def test_debian_builds_from_source(self, monkeypatch):
        monkeypatch.setattr(doc, "_os_release", lambda: {"ID": "ubuntu", "ID_LIKE": "debian"})
        steps = doc._install_modern_ydotool()
        joined = "\n".join(steps)
        assert "github.com/ReimuNotMoe/ydotool" in joined  # build from source
        assert "cmake" in joined

    def test_fedora_uses_package_manager(self, monkeypatch):
        monkeypatch.setattr(doc, "_os_release", lambda: {"ID": "fedora"})
        steps = doc._install_modern_ydotool()
        assert steps == ["sudo dnf install -y ydotool"]


class TestScreenshotCheck:
    def test_x11_mss_present(self, monkeypatch):
        monkeypatch.setattr(doc, "_session", lambda: "x11")
        monkeypatch.setattr(doc, "_module", lambda n: n == "mss")
        chk = doc._screenshot_check()
        assert chk.ok and "mss" in chk.detail

    def test_wayland_needs_grim(self, monkeypatch):
        monkeypatch.setattr(doc, "_session", lambda: "wayland")
        monkeypatch.setattr(doc, "_has", lambda b: None)
        monkeypatch.setattr(doc, "_os_release", lambda: {"ID": "ubuntu", "ID_LIKE": "debian"})
        chk = doc._screenshot_check()
        assert not chk.ok
        assert any("grim" in step for step in chk.fix)


class TestBuildReport:
    def test_sections_and_disabled_flag(self, monkeypatch):
        # Avoid touching the real ydotool / config.
        monkeypatch.setattr(doc, "_has", lambda b: None)
        monkeypatch.setattr(doc, "_module", lambda n: False)

        class _Cfg:
            class input_control:
                enabled = False

            class browser:
                enabled = False

            class local_llm:
                enabled = False

        sections = doc.build_report(_Cfg())
        titles = [s.title for s in sections]
        assert any("Desktop control" in t for t in titles)
        assert any("Browser" in t for t in titles)
        assert any("Voice" in t for t in titles)
        assert any("Local LLM" in t for t in titles)
        ic = next(s for s in sections if "Desktop control" in s.title)
        assert ic.enabled_now is False
