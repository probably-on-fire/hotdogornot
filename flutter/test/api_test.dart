// Tests for the Prediction / PredictResponse JSON parsers + helpers.
// The data-hygiene audit flagged that direct `as String` / `as num`
// casts in `fromJson` will throw uncaught TypeError on malformed
// responses — these tests pin the happy path and the obvious
// failure modes.

import 'package:flutter_test/flutter_test.dart';

import 'package:connector_id/src/api.dart';

void main() {
  group('Prediction.fromJson', () {
    test('parses a well-formed prediction', () {
      final p = Prediction.fromJson({
        'class_name': '2.4mm-M',
        'confidence': 0.83,
        'probabilities': {'SMA-M': 0.01, '2.4mm-M': 0.83},
        'bbox': {'x': 612, 'y': 415, 'w': 240, 'h': 240},
      });
      expect(p.className, '2.4mm-M');
      expect(p.confidence, closeTo(0.83, 1e-9));
      expect(p.probabilities['2.4mm-M'], closeTo(0.83, 1e-9));
      expect(p.bbox['x'], 612);
    });

    test('family/gender/isMale derive from class_name', () {
      final m = Prediction(
        className: '3.5mm-M',
        confidence: 0.9,
        probabilities: const {},
        bbox: const {},
      );
      expect(m.family, '3.5mm');
      expect(m.gender, 'M');
      expect(m.isMale, true);
      expect(m.isFemale, false);

      final f = Prediction(
        className: 'SMA-F',
        confidence: 0.5,
        probabilities: const {},
        bbox: const {},
      );
      expect(f.family, 'SMA');
      expect(f.gender, 'F');
      expect(f.isMale, false);
      expect(f.isFemale, true);
    });

    test('handles class_name without a hyphen gracefully', () {
      // Defensive: if the server ever sends a class without the M/F
      // suffix the parser shouldn't crash. family returns the whole
      // string, gender is empty, both isMale and isFemale are false.
      final p = Prediction(
        className: 'unknown',
        confidence: 0.1,
        probabilities: const {},
        bbox: const {},
      );
      expect(p.family, 'unknown');
      expect(p.gender, '');
      expect(p.isMale, false);
      expect(p.isFemale, false);
    });

    test('confidence accepts int (server may serialize 1 instead of 1.0)', () {
      final p = Prediction.fromJson(<String, dynamic>{
        'class_name': 'SMA-M',
        'confidence': 1,            // int, not double — must coerce
        'probabilities': <String, dynamic>{},
        'bbox': <String, dynamic>{},
      });
      expect(p.confidence, 1.0);
    });
  });

  group('PredictResponse.fromJson', () {
    test('parses the full response shape', () {
      final r = PredictResponse.fromJson({
        'image_width': 1920,
        'image_height': 1080,
        'predictions': [
          {
            'class_name': '2.4mm-M',
            'confidence': 0.83,
            'probabilities': {'2.4mm-M': 0.83},
            'bbox': {'x': 1, 'y': 2, 'w': 3, 'h': 4},
          },
        ],
      });
      expect(r.imageWidth, 1920);
      expect(r.imageHeight, 1080);
      expect(r.predictions, hasLength(1));
      expect(r.predictions.first.className, '2.4mm-M');
    });

    test('empty predictions list parses cleanly', () {
      final r = PredictResponse.fromJson({
        'image_width': 0,
        'image_height': 0,
        'predictions': const [],
      });
      expect(r.predictions, isEmpty);
    });

    test('extra unknown keys (like server _diag) are ignored', () {
      // The server attaches _diag fields per prediction for debugging.
      // The Flutter parser must tolerate them without throwing.
      final r = PredictResponse.fromJson({
        'image_width': 100,
        'image_height': 100,
        'predictions': [
          {
            'class_name': 'SMA-F',
            'confidence': 0.5,
            'probabilities': {'SMA-F': 0.5},
            'bbox': {'x': 0, 'y': 0, 'w': 10, 'h': 10},
            '_diag': {'fg_fraction': 0.3, 'center_density_ratio': 1.2},
          },
        ],
        'frames_scanned': 12,           // present on /predict-video
        'best_frame_index': 7,
      });
      expect(r.predictions, hasLength(1));
    });
  });
}
