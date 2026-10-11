"""Pure whitelist policy helpers."""

from __future__ import annotations

from collections.abc import Iterable


def normalize_ids(values: Iterable[object]) -> set[str]:
    return {str(value).strip() for value in values if str(value).strip()}


def is_allowed(
    *,
    user_id: str,
    group_id: str,
    is_group: bool,
    user_whitelist: Iterable[object],
    group_whitelist: Iterable[object],
    mode: str,
    allow_when_empty: bool,
    enable_group: bool,
    enable_private: bool,
) -> bool:
    """Evaluate QQ user/group whitelists without guessing missing identifiers."""

    if is_group and not enable_group:
        return False
    if not is_group and not enable_private:
        return False

    users = normalize_ids(user_whitelist)
    groups = normalize_ids(group_whitelist)
    checks: list[bool] = []
    if users:
        checks.append(str(user_id).strip() in users)
    if is_group and groups:
        checks.append(str(group_id).strip() in groups)

    if not checks:
        return bool(allow_when_empty)
    if str(mode).strip().lower() == "all":
        return all(checks)
    return any(checks)

