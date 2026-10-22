"""Position recovery for legacy weight-only, single-image training runs."""


def recovery_position(updates, epoch_size):
    if not isinstance(updates, int) or updates < 0 or epoch_size < 1:
        raise ValueError("Expected nonnegative update count and a nonempty epoch")
    completed, offset = divmod(updates, epoch_size)
    return completed + 1, offset


def recovery_offset(updates, epoch_size, epoch_count, batch_size, continuing, world_size=1):
    if updates < 0:
        raise ValueError("resume_total_iters must be nonnegative")
    if not updates:
        return 0  # Preserve the original behavior for other training entrypoints.
    if not continuing or batch_size != 1 or world_size != 1:
        raise ValueError("Update-count recovery requires continue_train, batch_size=1 and one process")
    expected_epoch, offset = recovery_position(updates, epoch_size)
    if epoch_count != expected_epoch:
        raise ValueError(f"Saved updates require epoch_count={expected_epoch}, received {epoch_count}")
    return offset
