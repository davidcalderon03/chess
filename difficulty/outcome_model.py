"""Load a trained checkpoint and predict win/draw/loss for the SIDE TO MOVE at ratings you choose.

    model = load_outcome_model("model.pt")
    pred = model.predict(samples, stm_rating=1600, opp_rating=1600)
    pred.win, pred.draw, pred.loss     # arrays (N,), side to move's point of view
    pred.defaults_used                 # e.g. {"white_clock_s": 600.0, ...} (empty for today's model)

`samples` is a SampleBatch(bb, meta) in the stored format of extract.py (see board_codec.py).

ADDING A NEW MODEL TYPE (e.g. one that also takes clocks, time control and history):
write ONE small class below, subclassing `ModelAdapter`:

    @register_adapter
    class ClockNetAdapter(ModelAdapter):
        name = "clocknet"
        uses_ratings = True
        required_context = ("white_clock_s", "black_clock_s", "time_control")   # defaults filled in
        @classmethod
        def matches(cls, ckpt): return ckpt.get("model_type") == "clocknet"
        @classmethod
        def from_checkpoint(cls, ckpt, device): ...build the net from the checkpoint, return cls(...)
        def predict_white_wdl(self, samples, white_elo, black_elo, context):
            ...return (probs (N,3) [white win, draw, black win], extras dict)

Everything else (side-to-move conversion, rating handling, default inputs, warnings, batching)
lives in OutcomeModel and does not change. The existing code (dataset.py / train.py) is imported,
never copied, so the input encoding and architecture stay in sync with teammates' updates.
"""
import warnings
from dataclasses import dataclass, field

import numpy as np

import config

LOUD = "\n" + "!" * 78 + "\n"


@dataclass
class SampleBatch:
    bb: np.ndarray    # (N, 12) uint64
    meta: np.ndarray  # (N, 7) uint8   [white_to_move, castling x4, ep_file, ply]

    def __len__(self):
        return len(self.meta)

    @property
    def white_to_move(self):
        return self.meta[:, 0].astype(bool)


@dataclass
class Prediction:
    win: np.ndarray    # side to move's point of view
    draw: np.ndarray
    loss: np.ndarray
    extras: dict = field(default_factory=dict)         # model-specific outputs (e.g. moves remaining)
    defaults_used: dict = field(default_factory=dict)  # inputs we filled with defaults -> value used
    rating_sensitive: bool = True                      # False = checkpoint ignores ratings
    model_kind: str = ""

    @property
    def score(self):
        return self.win + 0.5 * self.draw


class ModelAdapter:
    name = "base"
    uses_ratings = True
    required_context = ()

    @classmethod
    def matches(cls, ckpt):
        raise NotImplementedError

    @classmethod
    def from_checkpoint(cls, ckpt, device):
        raise NotImplementedError

    def predict_white_wdl(self, samples, white_elo, black_elo, context):
        """-> (probs (N,3) = [white win, draw, black win], extras dict)"""
        raise NotImplementedError


ADAPTERS = []


def register_adapter(cls):
    ADAPTERS.append(cls)
    return cls


@register_adapter
class ChessNetAdapter(ModelAdapter):
    """The current model: train.ChessNet over the 18x8x8 planes from dataset.LichessPositions."""
    name = "chessnet"
    required_context = ()

    def __init__(self, net, device, uses_ratings):
        self.net, self.device, self.uses_ratings = net, device, uses_ratings

    @classmethod
    def matches(cls, ckpt):
        sd = ckpt.get("model", {})
        return "stem.0.weight" in sd and "fc.0.weight" in sd

    @classmethod
    def from_checkpoint(cls, ckpt, device):
        from train import ChessNet  # imported, not copied
        a = ckpt.get("args", {})
        use_elo = bool(a.get("use_elo", ckpt["model"]["fc.0.weight"].shape[1] % 64 == 2))
        net = ChessNet(ch=a.get("channels", 128), blocks=a.get("blocks", 6), use_elo=use_elo)
        net.load_state_dict(ckpt["model"])
        return cls(net.to(device).eval(), device, use_elo)

    def predict_white_wdl(self, samples, white_elo, black_elo, context):
        import torch
        from dataset import LichessPositions  # imported, not copied: guarantees identical encoding
        n = len(samples)
        ds = LichessPositions.__new__(LichessPositions)  # shell object holding in-memory arrays
        ds.bb, ds.meta = samples.bb, samples.meta
        ds.elo = np.stack([white_elo, black_elo], 1).astype(np.int16)
        ds.y = np.zeros(n, np.uint8)
        out = []
        with torch.no_grad():
            for s in range(0, n, 1024):
                items = [ds[i] for i in range(s, min(n, s + 1024))]
                x = torch.stack([it[0] for it in items]).to(self.device)
                elo = torch.stack([it[1] for it in items]).to(self.device)
                out.append(torch.softmax(self.net(x, elo), dim=1).cpu().numpy())
        return np.concatenate(out).astype(np.float64), {}


class OutcomeModel:
    def __init__(self, adapter, checkpoint_path):
        self.adapter, self.checkpoint_path = adapter, str(checkpoint_path)
        if not adapter.uses_ratings:
            msg = (f"{LOUD}WARNING: checkpoint '{checkpoint_path}' was trained WITHOUT player ratings.\n"
                   f"Its predictions will NOT change with rating, so difficulty will be the same at every\n"
                   f"rating. Retrain with ratings (train.py --use-elo) for rating-dependent difficulty.{LOUD}")
            warnings.warn(msg, stacklevel=3)
            print(msg, flush=True)

    @property
    def rating_sensitive(self):
        return self.adapter.uses_ratings

    def predict(self, samples, stm_rating, opp_rating=None, context=None):
        """stm_rating / opp_rating: scalar or (N,) array. opp_rating defaults to stm_rating (same rating)."""
        n = len(samples)
        stm = np.broadcast_to(np.asarray(stm_rating, dtype=np.float64), (n,))
        opp = stm if opp_rating is None else np.broadcast_to(np.asarray(opp_rating, np.float64), (n,))
        wtm = samples.white_to_move
        white_elo, black_elo = np.where(wtm, stm, opp), np.where(wtm, opp, stm)
        ctx, used = {}, {}
        context = context or {}
        for key in self.adapter.required_context:
            if key in context:
                ctx[key] = context[key]
            else:
                ctx[key] = used[key] = config.DEFAULT_CONTEXT[key]
        probs, extras = self.adapter.predict_white_wdl(samples, white_elo, black_elo, ctx)
        pw, pd, pb = probs[:, 0], probs[:, 1], probs[:, 2]
        # Model outputs are WHITE's perspective; flip when black is to move.
        win, loss = np.where(wtm, pw, pb), np.where(wtm, pb, pw)
        return Prediction(win, pd, loss, extras, used, self.adapter.uses_ratings, self.adapter.name)


def load_outcome_model(path=None, device=None):
    import torch
    path = path or config.MODEL_PATH
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    for cls in ADAPTERS:
        if cls.matches(ckpt):
            dev = torch.device(device or "cpu")
            return OutcomeModel(cls.from_checkpoint(ckpt, dev), path)
    raise ValueError(f"no adapter recognizes checkpoint {path}; add one (see module docstring)")


def samples_from_boards(boards_and_plies):
    """Build a SampleBatch from [(chess.Board, ply), ...] using extract.encode."""
    import extract
    enc = [extract.encode(b, p) for b, p in boards_and_plies]
    return SampleBatch(np.stack([e[0] for e in enc]), np.stack([e[1] for e in enc]))
