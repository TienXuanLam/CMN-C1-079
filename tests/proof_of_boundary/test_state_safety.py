# PB-2 + PB-5: State Safety — CMN-C1-079 MultiLangIntentRoutingAgent
#
# Verifies:
#   PB-2  — State fields have no credential-like names
#   PB-5  — State fields use only msgpack-safe primitive types
#   PB-5d — State class is a TypedDict extending AgentState (runtime check)

import ast
import importlib
import os
import re

import pytest

_CREDENTIAL_FIELD_RE = re.compile(
    r"(?:^|_)(api_key|secret|password|credential|jwt|bearer|token)(?:_|$)",
    re.IGNORECASE,
)

_PROHIBITED_TYPE_ANNOTATIONS = [
    "BaseModel",
    "InvocationContext",
]

_LIVE_OBJECT_TYPE_NAMES = {
    "BaseModel",
    "dataclass",
}


def _state_file_path() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "schemas", "state.py"))


def _parse_state_file() -> ast.Module:
    with open(_state_file_path()) as f:
        return ast.parse(f.read(), filename=_state_file_path())


def _get_state_class(tree: ast.Module) -> ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "MultiLangIntentRoutingState":
            return node
    pytest.fail("MultiLangIntentRoutingState class not found in src/schemas/state.py")


def _field_names(cls: ast.ClassDef) -> list[tuple[str, ast.AnnAssign]]:
    return [
        (item.target.id, item)
        for item in cls.body
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
    ]


def _annotation_names(annotation: ast.expr) -> set[str]:
    names = set()
    for node in ast.walk(annotation):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


class TestNoCredentialFields:
    """PB-2: State must not store credential values."""

    def test_no_credential_like_field_names(self):
        tree = _parse_state_file()
        cls = _get_state_class(tree)
        violations = [
            f"  line {item.lineno}: {field_name}"
            for field_name, item in _field_names(cls)
            if _CREDENTIAL_FIELD_RE.search(field_name)
        ]
        assert violations == [], "Credential-like field names found in MultiLangIntentRoutingState:\n" + "\n".join(
            violations
        )


class TestMsgpackSafeTypes:
    """PB-5: All state field annotations must use only msgpack-safe types."""

    def test_no_prohibited_type_annotations(self):
        tree = _parse_state_file()
        cls = _get_state_class(tree)
        violations = []
        for field_name, item in _field_names(cls):
            if item.annotation:
                ann_names = _annotation_names(item.annotation)
                for prohibited in _PROHIBITED_TYPE_ANNOTATIONS:
                    if prohibited in ann_names:
                        violations.append(f"  line {item.lineno}: {field_name} — uses {prohibited}")
        assert violations == [], "Prohibited (non-msgpack-safe) types in MultiLangIntentRoutingState:\n" + "\n".join(
            violations
        )

    def test_no_live_object_types_in_annotations(self):
        tree = _parse_state_file()
        cls = _get_state_class(tree)
        violations = []
        for field_name, item in _field_names(cls):
            if item.annotation:
                found_live = _annotation_names(item.annotation) & _LIVE_OBJECT_TYPE_NAMES
                if found_live:
                    violations.append(f"  line {item.lineno}: {field_name} — {found_live}")
        assert violations == [], "Live object types in MultiLangIntentRoutingState:\n" + "\n".join(violations)


class TestRequiredDomainFields:
    """PB-5b: Required domain fields must be declared."""

    _REQUIRED = {
        "raw_message",
        "operator_config",
        "detected_language",
        "normalised_message",
        "intent_class",
        "intent_candidates",
        "confidence_score",
        "human_review_required",
        "urgency_level",
        "routing_target",
        "final_output",
        "error_code",
        "error_message",
    }

    def test_all_required_domain_fields_declared(self):
        tree = _parse_state_file()
        cls = _get_state_class(tree)
        declared = {name for name, _ in _field_names(cls)}
        missing = self._REQUIRED - declared
        assert not missing, f"Required fields missing from MultiLangIntentRoutingState: {sorted(missing)}"


class TestRuntimeImportSafety:
    """PB-5d: State class must be a TypedDict extending AgentState."""

    def test_state_is_typed_dict_not_pydantic(self):
        mod = importlib.import_module("src.schemas.state")
        StateClass = getattr(mod, "MultiLangIntentRoutingState")
        assert hasattr(
            StateClass, "__annotations__"
        ), "MultiLangIntentRoutingState has no __annotations__ — not a TypedDict?"
        try:
            from pydantic import BaseModel as PydanticBaseModel

            assert not issubclass(
                StateClass, PydanticBaseModel
            ), "MultiLangIntentRoutingState must not inherit from pydantic.BaseModel"
        except ImportError:
            pass

    def test_state_inherits_from_agent_state(self):
        mod = importlib.import_module("src.schemas.state")
        StateClass = getattr(mod, "MultiLangIntentRoutingState")
        from framework.schemas.agent_state import AgentState

        agent_state_keys = set(AgentState.__annotations__.keys())
        state_keys = set(StateClass.__annotations__.keys())
        missing = agent_state_keys - state_keys
        assert not missing, f"MultiLangIntentRoutingState is missing AgentState fields: {sorted(missing)}"
