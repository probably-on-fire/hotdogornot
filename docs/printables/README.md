# Printable scale markers

## `aruco_marker_25mm.png`

A 25 mm × 25 mm ArUco marker (dictionary `DICT_4X4_50`, ID `0`) on a white card with corner tick marks and a label.

### How to use

1. **Print** at 100% scale (no fit-to-page, no auto-shrink). The PNG embeds 600 DPI metadata so most printers will get this right by default.
2. **Verify the printed marker is 25 mm across** with a ruler. If it isn't, your printer scaled it — reprint with scaling explicitly disabled.
3. **Cut along the inner tick marks.** You want a small card you can place flat next to a connector during photography.
4. **Place the marker on the same surface as the connector**, in the same focal plane (don't lay it on a stack of papers raising it above the connector). The app uses it as a known-size reference to convert pixels to millimeters.

### Why this is needed

Without an in-frame scale reference, the measurement pipeline cannot reliably distinguish 2.92 mm vs 2.4 mm precision connectors — their aperture/hex pixel ratios are nearly identical, and recovering absolute size from a single uncalibrated photo is geometrically impossible. The ArUco marker provides absolute scale that resolves the ambiguity.

If you don't print this and use a 2.92 or 2.4 mm connector, the measurement will report `Unknown` (or guess between the two with low confidence) rather than commit to a wrong answer.
