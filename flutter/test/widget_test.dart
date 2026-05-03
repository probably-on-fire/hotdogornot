// Smoke test: app builds and shows the bottom-nav shell.
import 'package:flutter_test/flutter_test.dart';

import 'package:connector_id/src/app.dart';

void main() {
  testWidgets('app boots to bottom-nav shell', (WidgetTester tester) async {
    await tester.pumpWidget(const ConnectorIdApp());
    // Pump a few frames to let async settings.load() resolve.
    await tester.pump(const Duration(milliseconds: 50));
    // Three bottom-nav destinations.
    expect(find.text('Identify'), findsOneWidget);
    expect(find.text('Contribute'), findsOneWidget);
    expect(find.text('Settings'), findsOneWidget);
  });
}
