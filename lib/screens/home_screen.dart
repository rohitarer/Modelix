import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';
import 'MLModelScreen.dart'; // Import the MLModelScreen

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  late VideoPlayerController _controller;
  bool _isVideoInitialized = false; // Track video initialization status

  @override
  void initState() {
    super.initState();

    // Initialize video controller
    _controller = VideoPlayerController.asset('assets/images/page_1.mp4')
      ..initialize().then((_) {
        setState(() {
          _controller.setLooping(true);
          _controller.play();
          _isVideoInitialized = true; // Mark video as initialized
        });
      }).catchError((error) {
        debugPrint("Error initializing video: $error");
      });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          // Background video with opacity
          _isVideoInitialized
              ? SizedBox.expand(
                  child: Opacity(
                    opacity: 0.7, // Adjust opacity
                    child: FittedBox(
                      fit: BoxFit.cover,
                      child: SizedBox(
                        width: _controller.value.size.width,
                        height: _controller.value.size.height,
                        child: VideoPlayer(_controller),
                      ),
                    ),
                  ),
                )
              : const Center(
                  child: CircularProgressIndicator(), // Show loading indicator
                ),

          // Buttons Overlay
          Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                _buildButton("Regression Model", () {
                  _navigateToModelScreen(context, "Regression Model");
                }),
                const SizedBox(height: 20),
                _buildButton("Classification Model", () {
                  _navigateToModelScreen(context, "Classification Model");
                }),
                const SizedBox(height: 20),
                _buildButton("Clustering Model", () {
                  _navigateToModelScreen(context, "Clustering Model");
                }),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildButton(String label, VoidCallback onPressed) {
    return SizedBox(
      width: 250, // Fixed width for all buttons
      child: ElevatedButton(
        onPressed: onPressed,
        style: ElevatedButton.styleFrom(
          padding: const EdgeInsets.symmetric(vertical: 15),
          backgroundColor: Colors.black87, // Dark background color
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(20), // Rounded corners
          ),
          elevation: 8,
        ),
        child: Text(
          label,
          style: const TextStyle(
            fontSize: 18,
            fontWeight: FontWeight.bold,
            color: Colors.white, // White text
          ),
        ),
      ),
    );
  }

  void _navigateToModelScreen(BuildContext context, String modelType) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => MLModelScreen(modelType: modelType),
      ),
    );
  }
}
