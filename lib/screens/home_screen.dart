// import 'package:flutter/material.dart';
// import 'package:modelix/widgets/shining_text.dart';
// import 'package:video_player/video_player.dart';
// import 'MLModelScreen.dart'; // Import the MLModelScreen

// class HomeScreen extends StatefulWidget {
//   const HomeScreen({super.key});

//   @override
//   State<HomeScreen> createState() => _HomeScreenState();
// }

// class _HomeScreenState extends State<HomeScreen> with WidgetsBindingObserver {
//   late VideoPlayerController _controller;
//   bool _isInitialized = false;

//   @override
//   void initState() {
//     super.initState();
//     WidgetsBinding.instance.addObserver(this);
//     _initializeVideo();
//   }

//   void _initializeVideo() {
//     if (!_isInitialized) {
//       _controller = VideoPlayerController.asset('assets/images/page_1.mp4')
//         ..initialize().then((_) {
//           setState(() {
//             _isInitialized = true;
//             _controller.setLooping(true);
//             _controller.play();
//           });
//         }).catchError((error) {
//           debugPrint("Error initializing video: $error");
//         });
//     } else {
//       _controller.play();
//     }
//   }

//   @override
//   void didChangeAppLifecycleState(AppLifecycleState state) {
//     if (state == AppLifecycleState.resumed && _isInitialized) {
//       _controller.play();
//     } else if (state == AppLifecycleState.paused && _isInitialized) {
//       _controller.pause();
//     }
//   }

//   @override
//   void dispose() {
//     _controller.dispose();
//     WidgetsBinding.instance.removeObserver(this);
//     super.dispose();
//   }

//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       body: Stack(
//         children: [
//           // Background Video
//           _isInitialized
//               ? SizedBox.expand(
//                   child: Opacity(
//                     opacity: 0.8,
//                     child: FittedBox(
//                       fit: BoxFit.cover,
//                       child: SizedBox(
//                         width: _controller.value.size.width,
//                         height: _controller.value.size.height,
//                         child: VideoPlayer(_controller),
//                       ),
//                     ),
//                   ),
//                 )
//               : Center(
//                   child: Image.asset(
//                     'assets/images/loader.jpg',
//                     fit: BoxFit.cover,
//                     errorBuilder: (context, error, stackTrace) {
//                       return Container(
//                         color: Colors.black, // Fallback color
//                         child: const Center(
//                           child: Text(
//                             'Loading...',
//                             style: TextStyle(color: Colors.white, fontSize: 20),
//                           ),
//                         ),
//                       );
//                     },
//                   ),
//                 ),

//           // Foreground Content
//           Center(
//             child: Column(
//               mainAxisAlignment: MainAxisAlignment.center,
//               children: [
//                 const ShiningText(),
//                 const SizedBox(height: 20),
//                 _buildButton("Regression Model", () {
//                   _navigateToModelScreen(context, "Regression Model");
//                 }),
//                 const SizedBox(height: 20),
//                 _buildButton("Classification Model", () {
//                   _navigateToModelScreen(context, "Classification Model");
//                 }),
//                 const Text(
//                   '*Classification is under development*',
//                   style: TextStyle(color: Colors.red),
//                 ),
//                 const SizedBox(height: 10),
//                 _buildButton("Clustering Model", () {
//                   _navigateToModelScreen(context, "Clustering Model");
//                 }),
//                 const Text(
//                   '*Clustering is under development*',
//                   style: TextStyle(color: Colors.red),
//                 ),
//               ],
//             ),
//           ),
//         ],
//       ),
//     );
//   }

//   Widget _buildButton(String label, VoidCallback onPressed) {
//     return SizedBox(
//       width: 250,
//       child: ElevatedButton(
//         onPressed: onPressed,
//         style: ElevatedButton.styleFrom(
//           padding: const EdgeInsets.symmetric(vertical: 15),
//           backgroundColor: Colors.black87,
//           shape: RoundedRectangleBorder(
//             borderRadius: BorderRadius.circular(20),
//           ),
//           elevation: 8,
//         ),
//         child: Text(
//           label,
//           style: const TextStyle(
//             fontSize: 18,
//             fontWeight: FontWeight.bold,
//             color: Colors.white,
//           ),
//         ),
//       ),
//     );
//   }

//   void _navigateToModelScreen(BuildContext context, String modelType) {
//     Navigator.push(
//       context,
//       MaterialPageRoute(
//         builder: (context) => MLModelScreen(modelType: modelType),
//       ),
//     ).then((_) {
//       // Ensure video is playing when coming back
//       if (_controller.value.isInitialized) {
//         _controller.play();
//       }
//     });
//   }
// }

import 'package:flutter/material.dart';
import 'package:modelix/widgets/shining_text.dart';
import 'package:video_player/video_player.dart';
import 'MLModelScreen.dart'; // Import the MLModelScreen

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with WidgetsBindingObserver {
  late VideoPlayerController _controller;
  bool _isInitialized = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initializeVideo();
  }

  // void _initializeVideo() {
  //   print("Initializing video...");
  //   if (!_isInitialized) {
  //     _controller = VideoPlayerController.asset('assets/images/front.mp4')
  //       ..initialize().then((_) {
  //         setState(() {
  //           _isInitialized = true;
  //           _controller.setLooping(true);
  //           _controller.play();
  //         });
  //       }).catchError((error) {
  //         debugPrint("Error initializing video: $error");
  //       });
  //   } else {
  //     _controller.play();
  //   }
  // }

  void _initializeVideo() {
    print("Initializing video...");
    _controller = VideoPlayerController.asset('assets/images/front.mp4')
      ..initialize().then((_) {
        print("Video initialized successfully.");
        setState(() {
          _controller?.setLooping(true);
          _controller?.play();
        });
      }).catchError((error) {
        print("Error initializing video: $error");
      });
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed && _isInitialized) {
      _controller.play();
    } else if (state == AppLifecycleState.paused && _isInitialized) {
      _controller.pause();
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          // Background Video

          // _controller != null && _controller!.value.isInitialized
          //     ? Opacity(
          //         opacity: 0.8,
          //         child: SizedBox.expand(
          //           child: FittedBox(
          //             fit: BoxFit.cover,
          //             child: SizedBox(
          //               width: _controller!.value.size.width,
          //               height: _controller!.value.size.height,
          //               child: VideoPlayer(_controller!),
          //             ),
          //           ),
          //         ),
          //       )
          // :
          SizedBox.expand(
            child: Image.asset(
              'assets/images/loader2.jpg',
              fit: BoxFit.cover,
              errorBuilder: (context, error, stackTrace) {
                return Container(
                  color: Colors.black, // Fallback color
                  child: const Center(
                      // child: Text(
                      //   'Loading...',
                      //   style: TextStyle(color: Colors.white, fontSize: 20),
                      // ),
                      ),
                );
              },
            ),
          ),

          // Foreground Content
          Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const ShiningText(),
                const SizedBox(height: 20),
                _buildButton("Regression Model", () {
                  _navigateToModelScreen(context, "Regression Model", 1);
                }),
                const SizedBox(height: 20),
                _buildButton("Classification Model", () {
                  _navigateToModelScreen(context, "Classification Model", 2);
                }),
                const SizedBox(height: 20),
                _buildButton("Clustering Model", () {
                  _navigateToModelScreen(context, "Clustering Model", 3);
                }),
                const Text(
                  '*Clustering is under development*',
                  style: TextStyle(color: Colors.red),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildButton(String label, VoidCallback onPressed) {
    return SizedBox(
      width: 250,
      child: ElevatedButton(
        onPressed: onPressed,
        style: ElevatedButton.styleFrom(
          padding: const EdgeInsets.symmetric(vertical: 15),
          backgroundColor: Colors.black87,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(20),
          ),
          elevation: 8,
        ),
        child: Text(
          label,
          style: const TextStyle(
            fontSize: 18,
            fontWeight: FontWeight.bold,
            color: Colors.white,
          ),
        ),
      ),
    );
  }

  void _navigateToModelScreen(
      BuildContext context, String modelType, int mode) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => MLModelScreen(
          modelType: modelType,
          mode: mode, // Pass the mode to distinguish models
        ),
      ),
    ).then((_) {
      // Ensure video is playing when coming back
      if (_controller.value.isInitialized) {
        _controller.play();
      }
    });
  }
}
