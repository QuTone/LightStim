"""Private validation shared by code-family logical operation sets."""


def require_global_patch(builder, patch):
    """Require the patch view returned by this builder's QECSystem.add_patch."""
    uids = getattr(patch, "_registered_stabilizer_uids", None)
    for name, (registered, _) in builder.system.patches.items():
        if uids is not None and uids == registered._registered_stabilizer_uids:
            mapping = builder.system.local_to_global_map[name]
            expected = {mapping[q] for q in registered.data_indices}
            if patch.data_indices == expected:
                return
    raise ValueError("Pass the global patch returned by system.add_patch(), not a local patch.")
