import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:agrifusion_mobile/main.dart';

void main() {
  testWidgets('App starts successfully', (WidgetTester tester) async {
    await tester.pumpWidget(const AgriFusionApp());
    // Verify app renders
    expect(find.byType(MaterialApp), findsOneWidget);
  });
}
