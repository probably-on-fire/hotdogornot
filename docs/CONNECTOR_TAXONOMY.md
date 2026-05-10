# Connector Taxonomy

This taxonomy is the first stable target for RF connector identification.
The model should predict visual identity and attributes; connector
specifications should be looked up from structured data rather than memorized
by the neural network.

## Primary Families

| Family | Aliases | Notes |
|---|---|---|
| SMA | SubMiniature version A | Standard threaded 50 ohm RF connector, commonly used through about 18 GHz. |
| RP-SMA | Reverse-polarity SMA | SMA body/thread pattern with the center contact gender reversed. |
| 3.5mm | APC-3.5 style precision connector | Mechanically compatible with SMA, but higher precision and higher frequency. |
| 2.92mm | K, SMK | Precision connector mechanically compatible with SMA and 3.5mm, commonly used through about 40-46 GHz. |
| 2.4mm | none | Higher precision connector, not mechanically compatible with SMA/3.5mm/2.92mm. |
| 1.85mm | V | Millimeter-wave precision connector, commonly used through about 67 GHz. |
| 1.0mm | W | Very high frequency precision connector, commonly used through about 110 GHz. |
| SSMA | SubSubMiniature version A | Smaller SMA-related threaded connector. |
| SMB | SubMiniature B | Snap-on connector family, usually lower frequency than SMA. |
| SMC | SubMiniature C | Threaded subminiature connector, usually lower frequency than SMA. |
| QMA | Quick-lock SMA-like connector | Quick-lock/quick-disconnect alternative to SMA-style threaded coupling. |
| TNC | Threaded Neill-Concelman | Threaded RF connector used in antenna, cellular, and RF applications. |
| BNC | Bayonet Neill-Concelman | Bayonet quick-connect RF connector used in test, video, radio, and lab equipment. |
| MCX | Micro coaxial | Small snap-on connector for GPS, cellular, tuner, and compact RF hardware. |
| 7/16 DIN | DIN 7/16 | Large high-power threaded RF connector for antenna, cellular, and defense systems. |
| unknown | unsupported | First-class rejection or unsupported-family outcome. |

## Attribute Heads

The production model should not rely on a single flat class label. It should
emit independent attributes so the app can explain uncertainty and request
another view when needed.

| Attribute | Initial Values |
|---|---|
| presence | connector, no_connector, uncertain |
| family | SMA, RP-SMA, 3.5mm, 2.92mm, 2.4mm, 1.85mm, 1.0mm, SSMA, SMB, SMC, QMA, TNC, BNC, MCX, 7/16 DIN, unknown |
| precision_family | standard_sma, rp_sma, 3.5mm, 2.92mm_k_smk, 2.4mm, 1.85mm_v, 1.0mm_w, not_applicable, unknown |
| gender_contact | male_pin, female_socket, rp_male_body_female_contact, rp_female_body_male_contact, unknown |
| polarity | standard, reverse_polarity, not_applicable, unknown |
| mount_style | cable_mount, panel_mount, bulkhead, pcb_through_hole, pcb_edge_mount, pcb_surface_mount, adapter, terminator, unknown |
| orientation | straight, right_angle, tee, adapter_stack, unknown |
| cable_termination | solder, crimp, clamp, molded_cable, not_applicable, unknown |
| finish_material_cue | gold, nickel_silver, black_body, mixed, unknown |
| size_geometry | estimated_diameter, thread_count_or_pitch, body_length, connector_aperture, requires_calibrated_reference, unknown |
| confidence_state | high_confidence, ambiguous, insufficient_view, need_second_angle, need_scale_reference, unsupported_connector, no_connector_detected |

## Important Distinctions

SMA vs RP-SMA:

- Standard SMA male has a center pin.
- Standard SMA female has a center socket.
- RP-SMA reverses the center contact relative to the body/thread gender.
- Visual identification must look at both body/thread style and center contact.

SMA vs precision SMA-compatible connectors:

- SMA, 3.5mm, and 2.92mm can be mechanically compatible.
- Mechanical compatibility does not imply equal frequency performance.
- Visual-only classification may be ambiguous without scale, markings,
  manufacturer context, or a second angle.

Precision connector progression:

- SMA: general RF/microwave connector.
- 3.5mm: higher precision, SMA-compatible.
- 2.92mm/K/SMK: higher frequency, SMA/3.5mm-compatible.
- 2.4mm: higher precision, separate mating interface.
- 1.85mm/V: millimeter-wave precision connector.
- 1.0mm/W: very high frequency precision connector.

## Spec Lookup Rule

The neural network should infer the connector family and attributes. The app
or API should then cross-reference `training/rfconnectorai/specs/connectors.yaml`
for impedance, frequency range, coupling, compatibility, and visual notes.

This separation keeps the model focused on visible evidence and makes spec
updates auditable without retraining.

## Unknown and Abstention

`unknown` is not an error state. It is required behavior when:

- the connector is unsupported,
- the view is insufficient,
- the image contains no connector,
- two visually similar families cannot be separated confidently,
- a scale reference is required for size-sensitive identification.

For client-facing correctness, a low-confidence "need another angle" result
is better than a confident wrong identification.
