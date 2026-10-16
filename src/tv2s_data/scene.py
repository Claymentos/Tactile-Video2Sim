"""The Gemini scene file (scripts/reconstruction/semantic_parsing.py): helpers shared by the perception stages."""

SUPPORT_WORDS = ("table", "desk", "counter", "workbench", "surface")


def support_id(scene: dict) -> str:
    """The support surface the hands work on (a plane in the simulation, not reconstructed): the object named
    'table', else one whose id names a table / desk / counter / surface, else the first static one."""
    ids = [o["id"] for o in scene["objects"]]
    if "table" in ids:
        return "table"
    named = [i for i in ids if any(w in i for w in SUPPORT_WORDS)]
    static = [o["id"] for o in scene["objects"] if o.get("is_static")]
    return (named or static or ["table"])[0]
