from torch.utils.tensorboard.writer import SummaryWriter


writer = None  # TensorBoard SummaryWriter instance


def _log_scalars(scalars, step=0):
    for k, v in scalars.items():
        writer.add_scalar(k, v, step)


def on_pretrain_routine_start(trainer):
    global writer
    try:
        writer = SummaryWriter(str(trainer.save_dir))
    except Exception as e:
        print("Error initializing SummaryWriter:", e)


def on_batch_end(trainer):
    if writer is None:
        return
    _log_scalars(trainer.label_loss_items(trainer.tloss, prefix="train"), trainer.global_step)


def on_fit_epoch_end(trainer):
    if writer is None:
        return
    _log_scalars(trainer.metrics, trainer.epoch + 1)


callbacks = {
    "on_pretrain_routine_start": on_pretrain_routine_start,
    "on_fit_epoch_end": on_fit_epoch_end,
    "on_batch_end": on_batch_end
}

# Don't forget to close the SummaryWriter after training
if writer is not None:
    writer.close()
