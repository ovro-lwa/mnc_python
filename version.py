# -*- coding: utf-8 -*-
"""Version helpers without setuptools_scm dependency."""

from subprocess import Popen, PIPE
import re


def call_git_describe(abbrev):
    try:
        p = Popen(
            ['git', 'describe', '--abbrev=%d' % abbrev],
            stdout=PIPE,
            stderr=PIPE)
        p.stderr.close()
        line = p.stdout.readlines()[0]
        return line.strip()
    except Exception:
        return None


def is_dirty():
    try:
        p = Popen(
            ["git", "diff-index", "--name-only", "HEAD"],
            stdout=PIPE,
            stderr=PIPE)
        p.stderr.close()
        lines = p.stdout.readlines()
        return len(lines) > 0
    except Exception:
        return False


def read_release_version():
    try:
        with open("RELEASE-VERSION", "r") as f:
            return f.readlines()[0].strip()
    except Exception:
        return None


def write_release_version(version):
    with open("RELEASE-VERSION", "w") as f:
        f.write("%s\n" % version)


def pep440_from_describe(describe):
    if describe is None:
        return None

    describe = describe.lstrip("v")

    if re.match(r"^[0-9]+(?:\.[0-9]+)*$", describe):
        return describe

    parts = describe.rsplit("-", 2)
    if len(parts) == 3:
        tag, distance, gsha = parts
        if gsha.startswith("g") and distance.isdigit():
            tag = tag.lstrip("v")
            return "%s.dev%s+%s" % (tag, distance, gsha)

    normalized = re.sub(r"[^A-Za-z0-9\.]+", ".", describe).strip(".")
    return "0+%s" % normalized if normalized else "0"


def get_git_version(abbrev=7):
    release_version = read_release_version()

    describe = call_git_describe(abbrev)
    if describe is not None:
        describe = describe.decode("UTF-8")
    version = pep440_from_describe(describe)
    if version is not None and is_dirty():
        if "+" in version:
            version += ".dirty"
        else:
            version += "+dirty"

    if version is None:
        version = release_version

    if version is None:
        raise ValueError("Cannot find the version number!")

    if version != release_version:
        write_release_version(version)

    return version
