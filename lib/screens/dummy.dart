


// import 'package:flutter/material.dart';
// import 'package:video_player/video_player.dart';
// import 'package:file_picker/file_picker.dart';
// import 'dart:convert';
// import 'package:http/http.dart' as http;

// class MLModelScreen extends StatefulWidget {
//   final String modelType;

//   const MLModelScreen({super.key, required this.modelType});

//   @override
//   _MLModelScreenState createState() => _MLModelScreenState();
// }

// class _MLModelScreenState extends State<MLModelScreen> {
//   late VideoPlayerController _controller;
//   bool isLoading = false;
//   String? fileName;
//   List<String>? columns;
//   String xColumns = "";
//   String yColumn = "";
//   String? result;
//   String? codeTemplate;

//   @override
//   void initState() {
//     super.initState();
//     _initializeVideo();
//   }

//   void _initializeVideo() {
//     _controller = VideoPlayerController.asset('assets/images/page_1.mp4')
//       ..initialize().then((_) {
//         _controller.setLooping(true);
//         _controller.play();
//         setState(() {});
//       }).catchError((error) {
//         debugPrint("Error initializing video: $error");
//       });
//   }

//   @override
//   void dispose() {
//     _controller.dispose();
//     super.dispose();
//   }

//   Future<void> uploadCsvFile() async {
//     setState(() {
//       isLoading = true;
//       columns = null;
//       fileName = null;
//     });

//     try {
//       final pickedFile = await FilePicker.platform.pickFiles(
//         type: FileType.custom,
//         allowedExtensions: ['csv'],
//       );

//       if (pickedFile != null) {
//         final fileBytes = pickedFile.files.single.bytes;
//         final uploadedFileName = pickedFile.files.single.name;

//         final uri = Uri.parse('http://127.0.0.1:5000/get_columns');
//         final request = http.MultipartRequest('POST', uri);
//         request.files.add(
//           http.MultipartFile.fromBytes('file', fileBytes!,
//               filename: uploadedFileName),
//         );

//         final response = await request.send();
//         if (response.statusCode == 200) {
//           final responseData = await response.stream.bytesToString();
//           final data = json.decode(responseData);

//           setState(() {
//             fileName = data['file_name'];
//             columns = List<String>.from(data['columns']);
//             isLoading = false;
//           });
//         } else {
//           throw Exception("Failed to fetch columns.");
//         }
//       } else {
//         setState(() {
//           isLoading = false;
//         });
//       }
//     } catch (e) {
//       setState(() {
//         isLoading = false;
//       });
//       ScaffoldMessenger.of(context).showSnackBar(
//         SnackBar(content: Text('Error: ${e.toString()}')),
//       );
//     }
//   }

//   Future<void> processCsvFile() async {
//     if (xColumns.isEmpty || yColumn.isEmpty || fileName == null) {
//       ScaffoldMessenger.of(context).showSnackBar(
//         const SnackBar(
//           content:
//               Text('Please upload a CSV file and specify X and Y columns.'),
//         ),
//       );
//       return;
//     }

//     setState(() {
//       isLoading = true;
//       result = null;
//       codeTemplate = null;
//     });

//     try {
//       final uri = Uri.parse('http://127.0.0.1:5000/process_csv');
//       final response = await http.post(
//         uri,
//         headers: {'Content-Type': 'application/json'},
//         body: json.encode({
//           'file_name': fileName,
//           'x_columns': xColumns.split(','), // Split string into list
//           'y_column': yColumn,
//         }),
//       );

//       if (response.statusCode == 200) {
//         final data = json.decode(response.body);
//         setState(() {
//           result =
//               "Best Model: ${data['best_model']}\nAccuracy: ${data['accuracy']}";
//           codeTemplate = data['code_template'];
//           isLoading = false;
//         });
//       } else {
//         throw Exception("Failed to process file.");
//       }
//     } catch (e) {
//       setState(() {
//         isLoading = false;
//       });
//       ScaffoldMessenger.of(context).showSnackBar(
//         SnackBar(content: Text('Error: ${e.toString()}')),
//       );
//     }
//   }

//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       body: Stack(
//         children: [
//           // Background Video
//           _controller.value.isInitialized
//               ? Opacity(
//                   opacity: 0.8,
//                   child: SizedBox.expand(
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
//               : const Center(
//                   child: CircularProgressIndicator(),
//                 ),

//           // Foreground Content
//           Center(
//             child: SingleChildScrollView(
//               child: Container(
//                 padding: const EdgeInsets.all(20),
//                 margin: const EdgeInsets.all(16),
//                 decoration: BoxDecoration(
//                   color: Colors.black.withOpacity(0.7),
//                   borderRadius: BorderRadius.circular(16),
//                 ),
//                 child: Column(
//                   mainAxisSize: MainAxisSize.min,
//                   children: [
//                     Text(
//                       '${widget.modelType} Processor',
//                       style: const TextStyle(
//                         color: Colors.white,
//                         fontSize: 24,
//                         fontWeight: FontWeight.bold,
//                       ),
//                     ),
//                     const SizedBox(height: 20),
//                     ElevatedButton(
//                       onPressed: uploadCsvFile,
//                       child: const Text('Upload CSV File'),
//                     ),
//                     if (columns != null) ...[
//                       const SizedBox(height: 20),
//                       Text(
//                         'Available Columns: ${columns!.join(', ')}',
//                         style: const TextStyle(color: Colors.white),
//                       ),
//                       const SizedBox(height: 10),
//                       TextField(
//                         onChanged: (value) => xColumns = value,
//                         decoration: const InputDecoration(
//                           labelText: 'X Columns (comma-separated)',
//                           labelStyle: TextStyle(
//                             color: Color.fromARGB(255, 255, 106, 0),
//                             fontSize: 20,
//                             fontWeight: FontWeight.w500,
//                           ),
//                           floatingLabelBehavior: FloatingLabelBehavior.auto,
//                           border: OutlineInputBorder(),
//                           filled: true,
//                           fillColor: Color.fromARGB(255, 226, 226, 226),
//                         ),
//                       ),
//                       const SizedBox(height: 10),
//                       TextField(
//                         onChanged: (value) => yColumn = value,
//                         decoration: const InputDecoration(
//                           labelText: 'Y Column',
//                           labelStyle: TextStyle(
//                             color: Color.fromARGB(255, 255, 106, 0),
//                             fontSize: 20,
//                             fontWeight: FontWeight.w500,
//                           ),
//                           floatingLabelBehavior: FloatingLabelBehavior.auto,
//                           border: OutlineInputBorder(),
//                           filled: true,
//                           fillColor: Color.fromARGB(255, 226, 226, 226),
//                         ),
//                       ),
//                       const SizedBox(height: 20),
//                       ElevatedButton(
//                         onPressed: processCsvFile,
//                         child: const Text('Process CSV File'),
//                       ),
//                     ],
//                     if (result != null) ...[
//                       const SizedBox(height: 20),
//                       Text(
//                         result!,
//                         style: const TextStyle(color: Colors.green),
//                       ),
//                     ],
//                     if (codeTemplate != null) ...[
//                       const SizedBox(height: 20),
//                       const Text(
//                         'Generated Code:',
//                         style: TextStyle(
//                             fontWeight: FontWeight.bold, color: Colors.white),
//                       ),
//                       const SizedBox(height: 10),
//                       Text(
//                         codeTemplate!,
//                         style: const TextStyle(
//                             fontFamily: 'monospace', color: Colors.white),
//                       ),
//                     ],
//                   ],
//                 ),
//               ),
//             ),
//           ),
//         ],
//       ),
//     );
//   }
// }

// import 'package:flutter/material.dart';
// import 'package:video_player/video_player.dart';
// import 'package:file_picker/file_picker.dart';
// import 'dart:convert';
// import 'package:http/http.dart' as http;

// class MLModelScreen extends StatefulWidget {
//   final String modelType;

//   const MLModelScreen({super.key, required this.modelType});

//   @override
//   _MLModelScreenState createState() => _MLModelScreenState();
// }

// class _MLModelScreenState extends State<MLModelScreen> {
//   late VideoPlayerController _controller;
//   bool isLoading = false;
//   String? fileName;
//   List<String>? columns;
//   String xColumns = "";
//   String yColumn = "";
//   String? result;
//   String? codeTemplate;

//   final String backendUrl =
//       "http://127.0.0.1:55643/s4taRXKn-5U="; // Replace with your backend IP

//   @override
//   void initState() {
//     super.initState();
//     _initializeVideo();
//   }

//   void _initializeVideo() {
//     _controller = VideoPlayerController.asset('assets/images/page_1.mp4')
//       ..initialize().then((_) {
//         _controller.setLooping(true);
//         _controller.play();
//         setState(() {});
//       }).catchError((error) {
//         debugPrint("Error initializing video: $error");
//       });
//   }

//   @override
//   void dispose() {
//     _controller.dispose();
//     super.dispose();
//   }

//   Future<void> uploadCsvFile() async {
//     setState(() {
//       isLoading = true;
//       columns = null;
//       fileName = null;
//     });

//     try {
//       final pickedFile = await FilePicker.platform.pickFiles(
//         type: FileType.custom,
//         allowedExtensions: ['csv'],
//       );

//       if (pickedFile != null) {
//         final fileBytes = pickedFile.files.single.bytes;
//         final uploadedFileName = pickedFile.files.single.name;

//         final uri = Uri.parse('$backendUrl/get_columns');
//         debugPrint("Uploading file to $uri");

//         final request = http.MultipartRequest('POST', uri);
//         request.files.add(
//           http.MultipartFile.fromBytes('file', fileBytes!,
//               filename: uploadedFileName),
//         );

//         final response = await request.send();
//         final responseData = await response.stream.bytesToString();
//         debugPrint("Response: $responseData");

//         if (response.statusCode == 200) {
//           final data = json.decode(responseData);

//           setState(() {
//             fileName = data['file_name'];
//             columns = List<String>.from(data['columns']);
//             isLoading = false;
//           });
//         } else {
//           throw Exception("Failed to fetch columns.");
//         }
//       } else {
//         setState(() {
//           isLoading = false;
//         });
//       }
//     } catch (e) {
//       setState(() {
//         isLoading = false;
//       });
//       ScaffoldMessenger.of(context).showSnackBar(
//         SnackBar(content: Text('Error: ${e.toString()}')),
//       );
//     }
//   }

//   Future<void> processCsvFile() async {
//     if (xColumns.isEmpty || yColumn.isEmpty || fileName == null) {
//       ScaffoldMessenger.of(context).showSnackBar(
//         const SnackBar(
//           content:
//               Text('Please upload a CSV file and specify X and Y columns.'),
//         ),
//       );
//       return;
//     }

//     setState(() {
//       isLoading = true;
//       result = null;
//       codeTemplate = null;
//     });

//     try {
//       final uri = Uri.parse('$backendUrl/process_csv');
//       debugPrint("Processing file at $uri");

//       final response = await http.post(
//         uri,
//         headers: {'Content-Type': 'application/json'},
//         body: json.encode({
//           'file_name': fileName,
//           'x_columns': xColumns.split(','), // Split string into list
//           'y_column': yColumn,
//         }),
//       );

//       final responseData = response.body;
//       debugPrint("Response: $responseData");

//       if (response.statusCode == 200) {
//         final data = json.decode(response.body);
//         setState(() {
//           result =
//               "Best Model: ${data['best_model']}\nAccuracy: ${data['accuracy']}";
//           codeTemplate = data['code_template'];
//           isLoading = false;
//         });
//       } else {
//         throw Exception("Failed to process file.");
//       }
//     } catch (e) {
//       setState(() {
//         isLoading = false;
//       });
//       ScaffoldMessenger.of(context).showSnackBar(
//         SnackBar(content: Text('Error: ${e.toString()}')),
//       );
//     }
//   }

//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       body: Stack(
//         children: [
//           _controller.value.isInitialized
//               ? Opacity(
//                   opacity: 0.8,
//                   child: SizedBox.expand(
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
//               : const Center(
//                   child: CircularProgressIndicator(),
//                 ),
//           Center(
//             child: SingleChildScrollView(
//               child: Container(
//                 padding: const EdgeInsets.all(20),
//                 margin: const EdgeInsets.all(16),
//                 decoration: BoxDecoration(
//                   color: Colors.black.withOpacity(0.7),
//                   borderRadius: BorderRadius.circular(16),
//                 ),
//                 child: Column(
//                   mainAxisSize: MainAxisSize.min,
//                   children: [
//                     Text(
//                       '${widget.modelType} Processor',
//                       style: const TextStyle(
//                         color: Colors.white,
//                         fontSize: 24,
//                         fontWeight: FontWeight.bold,
//                       ),
//                     ),
//                     const SizedBox(height: 20),
//                     ElevatedButton(
//                       onPressed: uploadCsvFile,
//                       child: const Text('Upload CSV File'),
//                     ),
//                     if (columns != null) ...[
//                       const SizedBox(height: 20),
//                       Text(
//                         'Available Columns: ${columns!.join(', ')}',
//                         style: const TextStyle(color: Colors.white),
//                       ),
//                       const SizedBox(height: 10),
//                       TextField(
//                         onChanged: (value) => xColumns = value,
//                         decoration: const InputDecoration(
//                           labelText: 'X Columns (comma-separated)',
//                         ),
//                       ),
//                       const SizedBox(height: 10),
//                       TextField(
//                         onChanged: (value) => yColumn = value,
//                         decoration: const InputDecoration(
//                           labelText: 'Y Column',
//                         ),
//                       ),
//                       const SizedBox(height: 20),
//                       ElevatedButton(
//                         onPressed: processCsvFile,
//                         child: const Text('Process CSV File'),
//                       ),
//                     ],
//                     if (result != null) ...[
//                       const SizedBox(height: 20),
//                       Text(
//                         result!,
//                         style: const TextStyle(color: Colors.green),
//                       ),
//                     ],
//                     if (codeTemplate != null) ...[
//                       const SizedBox(height: 20),
//                       const Text(
//                         'Generated Code:',
//                         style: TextStyle(
//                             fontWeight: FontWeight.bold, color: Colors.white),
//                       ),
//                       const SizedBox(height: 10),
//                       Text(
//                         codeTemplate!,
//                         style: const TextStyle(
//                             fontFamily: 'monospace', color: Colors.white),
//                       ),
//                     ],
//                   ],
//                 ),
//               ),
//             ),
//           ),
//         ],
//       ),
//     );
//   }
// }
