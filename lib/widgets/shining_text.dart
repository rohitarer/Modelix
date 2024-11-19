import 'package:flutter/material.dart';

class ShiningText extends StatefulWidget {
  const ShiningText({super.key});

  @override
  _ShiningTextState createState() => _ShiningTextState();
}

class _ShiningTextState extends State<ShiningText>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 3),
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        return ShaderMask(
          shaderCallback: (bounds) {
            return LinearGradient(
              colors: const [
                Colors.yellow,
                Colors.orange,
                Colors.red,
                Colors.purple,
                Colors.blue,
              ],
              stops: [
                _controller.value - 0.2,
                _controller.value - 0.1,
                _controller.value,
                _controller.value + 0.1,
                _controller.value + 0.2,
              ].map((e) => e.clamp(0.0, 1.0)).toList(),
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ).createShader(bounds);
          },
          blendMode: BlendMode.srcIn,
          child: const Text(
            'Modelix',
            style: TextStyle(
              fontSize: 100,
              fontWeight: FontWeight.bold,
              color: Colors.black, // Fallback color (if shader fails)
            ),
          ),
        );
      },
    );
  }
}
