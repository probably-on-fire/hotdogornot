# Demo Script

A 5-10 minute walk-through of the connector identification system. Use
this as a script the first time, then adapt to the audience.

## 1. Frame The Problem (60 seconds)

> "Identifying an RF connector from a glance is hard. Even experienced RF
> engineers regularly confuse SMA with 3.5mm or 2.92mm — they thread
> together but they are *not* electrically equivalent. RP-SMA looks like
> SMA but the gender of the center contact is reversed."

> "Connector ID identifies the family and the right details — gender,
> polarity, mount style, orientation, termination — and it cross-references
> the connector spec. It tells you when it is not sure."

## 2. Live Identification (2-3 minutes)

1. Start the FastAPI server (`uvicorn rfconnectorai.server.predict_service:app --port 8503`).
2. Run the Flutter app and connect it to the server.
3. Pick up an SMA male cable. Aim camera. Identify.
4. Replace with an RP-SMA cable. Identify. Show the polarity flip.
5. Pick up an adapter. Identify. Highlight the side A / side B blocks in
   the result card.
6. Show a confusing object (a 3.5mm headphone jack). Identify and show
   the abstention or low-confidence state.

## 3. Show The Honest States (1 minute)

Trigger each abstention state intentionally:

- ``need_second_angle``: hold the connector at an angle that hides the
  contact.
- ``need_scale_reference``: hold a connector that requires geometry
  verification, with no ruler in frame.
- ``unsupported_connector``: aim at something completely off-domain.
- ``no_connector_detected``: aim at a blank desk.

> "Forced wrong guesses are unacceptable. The model is allowed to ask
> for help."

## 4. Cross-Reference The Spec (1 minute)

After a successful identification, expand the result card. The spec is
joined from ``training/rfconnectorai/specs/connectors.yaml``, not memorized
by the neural network — point this out: it is updateable without
retraining the model.

## 5. Show The Acceptance Gates (1 minute)

Open [`ACCEPTANCE_GATES.md`](ACCEPTANCE_GATES.md). Walk through G0..G5.
Identify which gate is currently passing and which gate the next batch is
working toward.

## 6. Address The Hard Question (1 minute)

> "We are not claiming 99.99% accuracy yet. The current real-phone
> holdout is 8 images, which is too small to support strong claims. The
> roadmap to credible high-accuracy claims runs through:
>
> 1. A larger real-phone holdout (G3 / G4 in the implementation plan).
> 2. Geometry / scale verification with a printed scale marker.
> 3. Optional second-angle capture for ambiguous cases.
> 4. Abstention-aware metrics, not forced-choice accuracy."

End on what is on the immediate roadmap (next gate, next batch). Avoid
selling a future feature as if it ships today.

## After The Demo

- Note any false positives, false negatives, or surprising failures.
- Add the failure cases to the contributor flow so they enter the next
  training round.
- Open [`LIMITATIONS_AND_NEXT_STEPS.md`](LIMITATIONS_AND_NEXT_STEPS.md)
  and walk the client through the next two batches.
