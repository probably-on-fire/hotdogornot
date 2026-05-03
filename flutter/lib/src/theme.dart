import 'package:flutter/material.dart';

/// Dark theme matching the rest of the project (the labeler uses similar
/// dark grays + a blue accent).
ThemeData buildAppTheme() {
  const surface = Color(0xFF161A22);
  const surfaceAlt = Color(0xFF1D222C);
  const accent = Color(0xFF4F8CFF);
  const text = Color(0xFFD6DDE8);
  const muted = Color(0xFF8A93A3);

  final base = ThemeData.dark(useMaterial3: true);
  return base.copyWith(
    scaffoldBackgroundColor: const Color(0xFF0F1115),
    colorScheme: base.colorScheme.copyWith(
      primary: accent,
      secondary: accent,
      surface: surface,
      onSurface: text,
    ),
    appBarTheme: const AppBarTheme(
      backgroundColor: surface,
      elevation: 0,
      titleTextStyle: TextStyle(
        color: text,
        fontSize: 17,
        fontWeight: FontWeight.w600,
      ),
    ),
    cardTheme: CardThemeData(
      color: surface,
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: const BorderSide(color: Color(0xFF2A313E)),
      ),
    ),
    elevatedButtonTheme: ElevatedButtonThemeData(
      style: ElevatedButton.styleFrom(
        backgroundColor: accent,
        foregroundColor: Colors.white,
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(8),
        ),
        textStyle: const TextStyle(fontSize: 15, fontWeight: FontWeight.w600),
      ),
    ),
    outlinedButtonTheme: OutlinedButtonThemeData(
      style: OutlinedButton.styleFrom(
        foregroundColor: text,
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
        side: const BorderSide(color: Color(0xFF2A313E)),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(8),
        ),
        textStyle: const TextStyle(fontSize: 15, fontWeight: FontWeight.w600),
      ),
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: surfaceAlt,
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(8),
        borderSide: const BorderSide(color: Color(0xFF2A313E)),
      ),
      labelStyle: const TextStyle(color: muted),
    ),
    textTheme: base.textTheme.apply(
      bodyColor: text,
      displayColor: text,
    ),
  );
}

/// Hot-dog blue (M)
const kHotDogColor = Color(0xFFFF6B35);

/// Not-hot-dog teal (F)
const kNotHotDogColor = Color(0xFF00A6A6);
