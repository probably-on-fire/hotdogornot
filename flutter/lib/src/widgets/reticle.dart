import 'dart:math' as math;

import 'package:flutter/material.dart';

/// Shared centered reticle (target circle + corner ticks + crosshair)
/// used on both the Identify and Contribute screens. The user fits the
/// connector mating face inside the ring; the camera screen crops the
/// captured photo to a square inscribing the ring before upload, giving
/// the classifier a consistent scale regardless of phone or distance.
class ReticleOverlay extends StatelessWidget {
  const ReticleOverlay({super.key, this.hint = 'CENTER CONNECTOR'});
  final String hint;

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _ReticlePainter(hint: hint),
      child: const SizedBox.expand(),
    );
  }
}

class _ReticlePainter extends CustomPainter {
  _ReticlePainter({required this.hint});
  final String hint;

  @override
  void paint(Canvas canvas, Size size) {
    final cx = size.width / 2;
    final cy = size.height / 2;
    final r = math.min(size.width, size.height) * 0.28;
    final ringPaint = Paint()
      ..color = Colors.white.withOpacity(0.55)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.5;
    final dashPaint = Paint()
      ..color = Colors.white.withOpacity(0.85)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0;
    canvas.drawCircle(Offset(cx, cy), r, ringPaint);
    final tick = r * 0.18;
    final s = r / math.sqrt(2);
    for (final sx in const [-1.0, 1.0]) {
      for (final sy in const [-1.0, 1.0]) {
        final ax = cx + sx * s;
        final ay = cy + sy * s;
        canvas.drawLine(Offset(ax, ay), Offset(ax - sx * tick, ay), dashPaint);
        canvas.drawLine(Offset(ax, ay), Offset(ax, ay - sy * tick), dashPaint);
      }
    }
    final ch = r * 0.08;
    canvas.drawLine(Offset(cx - ch, cy), Offset(cx + ch, cy), dashPaint);
    canvas.drawLine(Offset(cx, cy - ch), Offset(cx, cy + ch), dashPaint);
    final tp = TextPainter(
      text: TextSpan(
        text: hint,
        style: TextStyle(
          color: Colors.white.withOpacity(0.7),
          fontSize: 11,
          fontWeight: FontWeight.w700,
          letterSpacing: 1.5,
        ),
      ),
      textDirection: TextDirection.ltr,
    );
    tp.layout();
    tp.paint(canvas, Offset(cx - tp.width / 2, cy - r - tp.height - 8));
  }

  @override
  bool shouldRepaint(covariant _ReticlePainter old) => old.hint != hint;
}
