// Smoke test: app builds and shows the home screen.
import 'package:flutter_test/flutter_test.dart';

import 'package:connector_id/src/app.dart';

void main() {
  testWidgets('app boots to home screen', (WidgetTester tester) async {
    await tester.pumpWidget(const ConnectorIdApp());
    // Pump a few frames to let async settings.load() resolve.
    await tester.pump(const Duration(milliseconds: 50));
    expect(find.text('Identify a connector'), findsOneWidget);
    expect(find.text('Contribute training data'), findsOneWidget);
  });
}
