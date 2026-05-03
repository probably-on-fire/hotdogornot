// Smoke test: app builds and shows the bottom-nav shell.
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:connector_id/src/app.dart';

void main() {
  setUp(() {
    SharedPreferences.setMockInitialValues({});
  });

  testWidgets('app boots to bottom-nav shell', (WidgetTester tester) async {
    await tester.pumpWidget(const ConnectorIdApp());
    // Give Settings.load() (one microtask) time to resolve. Can't use
    // pumpAndSettle here because the Identify tab kicks off a camera-init
    // future that never resolves in the headless test environment.
    for (int i = 0; i < 10; i++) {
      await tester.pump(const Duration(milliseconds: 100));
      if (find.text('Identify').evaluate().isNotEmpty) break;
    }
    expect(find.text('Identify'), findsOneWidget);
    expect(find.text('Contribute'), findsOneWidget);
    expect(find.text('Settings'), findsOneWidget);
  });
}
