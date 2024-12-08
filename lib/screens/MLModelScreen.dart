import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';
import 'package:file_picker/file_picker.dart';
import 'package:pdf/widgets.dart' as pw;
import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'dart:html' as html; // Web-specific for file downloads

class MLModelScreen extends StatefulWidget {
  final String modelType;
  final int mode; // Add the mode parameter

  const MLModelScreen({super.key, required this.modelType, required this.mode});

  @override
  _MLModelScreenState createState() => _MLModelScreenState();
}

class _MLModelScreenState extends State<MLModelScreen>
    with WidgetsBindingObserver {
  VideoPlayerController? _controller;
  bool isLoading = false;
  String? fileName;
  List<String>? columns;
  String xColumns = "";
  String yColumn = "";
  String? result;
  String? codeTemplate;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _initializeVideo();
    });
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _controller?.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.inactive && _controller != null) {
      _controller!.pause();
    } else if (state == AppLifecycleState.resumed && _controller != null) {
      _controller!.play();
    }
  }

  void _initializeVideo() {
    print("Initializing video...");
    _controller = VideoPlayerController.asset('assets/images/back.mp4')
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

  void downloadPdfWeb(String fileName, List<int> pdfBytes) {
    final blob = html.Blob([pdfBytes]);
    final url = html.Url.createObjectUrlFromBlob(blob);
    final anchor = html.AnchorElement(href: url)
      ..target = 'blank'
      ..download = fileName;
    anchor.click();
    html.Url.revokeObjectUrl(url);
  }

  Future<void> uploadCsvFile() async {
    setState(() {
      isLoading = true;
      columns = null;
      fileName = null;
    });

    try {
      final pickedFile = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['csv'],
      );

      if (pickedFile != null) {
        final fileBytes = pickedFile.files.single.bytes;
        final uploadedFileName = pickedFile.files.single.name;

        final uri = Uri.parse('http://127.0.0.1:5000/get_columns');
        print("Attempting to send file to: $uri");
        final request = http.MultipartRequest('POST', uri);
        request.files.add(
          http.MultipartFile.fromBytes('file', fileBytes!,
              filename: uploadedFileName),
        );

        final response = await request.send();
        print("Response status code: ${response.statusCode}");
        if (response.statusCode == 200) {
          final responseData = await response.stream.bytesToString();
          final data = json.decode(responseData);
          print("Response data: $data");

          setState(() {
            fileName = data['file_name'];
            columns = List<String>.from(data['columns']);
            isLoading = false;
          });
        } else {
          throw Exception("Failed to fetch columns.");
        }
      } else {
        setState(() {
          isLoading = false;
        });
        print("No file selected.");
      }
    } catch (e) {
      setState(() {
        isLoading = false;
      });
      print("Error: $e");
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: ${e.toString()}')),
      );
    }
  }

  Future<void> processCsvFile() async {
    if (xColumns.isEmpty || yColumn.isEmpty || fileName == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content:
              Text('Please upload a CSV file and specify X and Y columns.'),
        ),
      );
      return;
    }

    setState(() {
      isLoading = true;
      result = null;
      codeTemplate = null;
    });

    try {
      final uri = Uri.parse('http://127.0.0.1:5000/process_csv');
      final response = await http.post(
        uri,
        headers: {'Content-Type': 'application/json'},
        body: json.encode({
          'file_name': fileName,
          'x_columns': xColumns.split(','), // Split string into list
          'y_column': yColumn,
          'mode': widget.mode, // Pass the mode to the backend
        }),
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        setState(() {
          result =
              "Best Model: ${data['best_model']}\nAccuracy: ${data['accuracy']}";
          codeTemplate = data['code_template'];
          isLoading = false;
        });
      } else {
        throw Exception("Failed to process file.");
      }
    } catch (e) {
      setState(() {
        isLoading = false;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: ${e.toString()}')),
      );
    }
  }

  Future<void> downloadPDF() async {
    if (codeTemplate == null) return;

    try {
      final pdf = pw.Document();
      pdf.addPage(
        pw.Page(
          build: (pw.Context context) => pw.Center(
            child: pw.Text(
              codeTemplate!,
              style: const pw.TextStyle(fontSize: 12),
            ),
          ),
        ),
      );

      // Save PDF for Web
      final pdfBytes = await pdf.save();
      downloadPdfWeb("generated_code.pdf", pdfBytes);
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('PDF downloaded')),
      );
    } catch (e) {
      print("Error saving PDF: $e");
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error saving PDF: $e')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          // Background Video
          _controller != null && _controller!.value.isInitialized
              ? Opacity(
                  opacity: 0.9,
                  child: SizedBox.expand(
                    child: FittedBox(
                      fit: BoxFit.cover,
                      child: SizedBox(
                        width: _controller!.value.size.width,
                        height: _controller!.value.size.height,
                        child: VideoPlayer(_controller!),
                      ),
                    ),
                  ),
                )
              : const Center(
                  child: CircularProgressIndicator(),
                ),

          // Foreground Content
          Center(
            child: SingleChildScrollView(
              child: Container(
                padding: const EdgeInsets.all(20),
                margin: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.7),
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      '${widget.modelType} Processor',
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 20),
                    ElevatedButton(
                      onPressed: uploadCsvFile,
                      child: const Text('Upload CSV File'),
                    ),
                    if (columns != null) ...[
                      const SizedBox(height: 20),
                      Text(
                        'Available Columns: ${columns!.join(', ')}',
                        style: const TextStyle(color: Colors.white),
                      ),
                      const SizedBox(height: 10),
                      TextField(
                        onChanged: (value) => xColumns = value,
                        decoration: const InputDecoration(
                          labelText: 'X Columns (comma-separated)',
                          border: OutlineInputBorder(),
                          filled: true,
                          fillColor: Colors.white70,
                        ),
                      ),
                      const SizedBox(height: 10),
                      TextField(
                        onChanged: (value) => yColumn = value,
                        decoration: const InputDecoration(
                          labelText: 'Y Column',
                          border: OutlineInputBorder(),
                          filled: true,
                          fillColor: Colors.white70,
                        ),
                      ),
                      const SizedBox(height: 20),
                      ElevatedButton(
                        onPressed: processCsvFile,
                        child: const Text('Process CSV File'),
                      ),
                    ],
                    if (result != null) ...[
                      const SizedBox(height: 20),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          const Spacer(),
                          const Text(
                            'Generated Code:',
                            style: TextStyle(
                              fontWeight: FontWeight.bold,
                              color: Colors.white,
                            ),
                          ),
                          const Spacer(),
                          IconButton(
                            icon: Image.asset(
                              'assets/images/pdf.png', // Path to your image
                              height: 40, // Set height
                              width: 40, // Set width
                            ),
                            onPressed:
                                downloadPDF, // Call your download function
                          ),
                        ],
                      ),
                      const SizedBox(height: 10),
                      Text(
                        codeTemplate!,
                        style: const TextStyle(
                          fontFamily: 'monospace',
                          color: Colors.white,
                        ),
                      ),
                    ],
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}





// import 'package:flutter/material.dart';
// import 'package:video_player/video_player.dart';
// import 'package:file_picker/file_picker.dart';
// import 'package:pdf/pdf.dart';
// import 'package:pdf/widgets.dart' as pw;
// import 'package:path_provider/path_provider.dart';
// import 'dart:convert';
// import 'dart:io';
// import 'package:http/http.dart' as http;
// import 'dart:html' as html;

// class MLModelScreen extends StatefulWidget {
//   final String modelType;
//   final int mode; // Add the mode parameter

//   const MLModelScreen({super.key, required this.modelType, required this.mode});

//   @override
//   _MLModelScreenState createState() => _MLModelScreenState();
// }

// class _MLModelScreenState extends State<MLModelScreen>
//     with WidgetsBindingObserver {
//   VideoPlayerController? _controller;
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
//     WidgetsBinding.instance.addObserver(this);
//     WidgetsBinding.instance.addPostFrameCallback((_) {
//       _initializeVideo();
//     });
//   }

//   @override
//   void dispose() {
//     WidgetsBinding.instance.removeObserver(this);
//     _controller?.dispose();
//     super.dispose();
//   }

//   @override
//   void didChangeAppLifecycleState(AppLifecycleState state) {
//     if (state == AppLifecycleState.inactive && _controller != null) {
//       _controller!.pause();
//     } else if (state == AppLifecycleState.resumed && _controller != null) {
//       _controller!.play();
//     }
//   }

//   void _initializeVideo() {
//     print("Initializing video...");
//     _controller = VideoPlayerController.asset('assets/images/page_1.mp4')
//       ..initialize().then((_) {
//         print("Video initialized successfully.");
//         setState(() {
//           _controller?.setLooping(true);
//           _controller?.play();
//         });
//       }).catchError((error) {
//         print("Error initializing video: $error");
//       });
//   }

//   void downloadPdfWeb(String fileName, List<int> pdfBytes) {
//     final blob = html.Blob([pdfBytes]);
//     final url = html.Url.createObjectUrlFromBlob(blob);
//     final anchor = html.AnchorElement(href: url)
//       ..target = 'blank'
//       ..download = fileName;
//     anchor.click();
//     html.Url.revokeObjectUrl(url);
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
//         print("Attempting to send file to: $uri");
//         final request = http.MultipartRequest('POST', uri);
//         request.files.add(
//           http.MultipartFile.fromBytes('file', fileBytes!,
//               filename: uploadedFileName),
//         );

//         final response = await request.send();
//         print("Response status code: ${response.statusCode}");
//         if (response.statusCode == 200) {
//           final responseData = await response.stream.bytesToString();
//           final data = json.decode(responseData);
//           print("Response data: $data");

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
//         print("No file selected.");
//       }
//     } catch (e) {
//       setState(() {
//         isLoading = false;
//       });
//       print("Error: $e");
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
//           'mode': widget.mode, // Pass the mode to the backend
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

//   Future<void> downloadPDF() async {
//     if (codeTemplate == null) return;

//     try {
//       final pdf = pw.Document();
//       pdf.addPage(
//         pw.Page(
//           build: (pw.Context context) => pw.Center(
//             child: pw.Text(
//               codeTemplate!,
//               style: const pw.TextStyle(fontSize: 12),
//             ),
//           ),
//         ),
//       );

//       // Get the directory to save the file
//       final directory = await getTemporaryDirectory();
//       final file = File('${directory.path}/generated_code.pdf');
//       await file.writeAsBytes(await pdf.save());

//       ScaffoldMessenger.of(context).showSnackBar(
//         SnackBar(content: Text('PDF downloaded to ${file.path}')),
//       );
//     } catch (e) {
//       print("Error saving PDF: $e");
//       ScaffoldMessenger.of(context).showSnackBar(
//         SnackBar(content: Text('Error saving PDF: $e')),
//       );
//     }
//   }

//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       body: Stack(
//         children: [
//           // Background Video
//           _controller != null && _controller!.value.isInitialized
//               ? Opacity(
//                   opacity: 0.8,
//                   child: SizedBox.expand(
//                     child: FittedBox(
//                       fit: BoxFit.cover,
//                       child: SizedBox(
//                         width: _controller!.value.size.width,
//                         height: _controller!.value.size.height,
//                         child: VideoPlayer(_controller!),
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
//                           border: OutlineInputBorder(),
//                           filled: true,
//                           fillColor: Colors.white70,
//                         ),
//                       ),
//                       const SizedBox(height: 10),
//                       TextField(
//                         onChanged: (value) => yColumn = value,
//                         decoration: const InputDecoration(
//                           labelText: 'Y Column',
//                           border: OutlineInputBorder(),
//                           filled: true,
//                           fillColor: Colors.white70,
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
//                       Row(
//                         mainAxisAlignment: MainAxisAlignment.spaceBetween,
//                         children: [
//                           const Text(
//                             'Generated Code:',
//                             style: TextStyle(
//                               fontWeight: FontWeight.bold,
//                               color: Colors.white,
//                             ),
//                           ),
//                           IconButton(
//                             icon: const Icon(Icons.download),
//                             color: Colors.white,
//                             onPressed: downloadPDF,
//                           ),
//                         ],
//                       ),
//                       const SizedBox(height: 10),
//                       Text(
//                         codeTemplate!,
//                         style: const TextStyle(
//                           fontFamily: 'monospace',
//                           color: Colors.white,
//                         ),
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
//   final int mode; // Add the mode parameter

//   const MLModelScreen({super.key, required this.modelType, required this.mode});

//   @override
//   _MLModelScreenState createState() => _MLModelScreenState();
// }

// class _MLModelScreenState extends State<MLModelScreen>
//     with WidgetsBindingObserver {
//   VideoPlayerController? _controller;
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
//     WidgetsBinding.instance.addObserver(this);
//     WidgetsBinding.instance.addPostFrameCallback((_) {
//       _initializeVideo();
//     });
//   }

//   @override
//   void dispose() {
//     WidgetsBinding.instance.removeObserver(this);
//     _controller?.dispose();
//     super.dispose();
//   }

//   @override
//   void didChangeAppLifecycleState(AppLifecycleState state) {
//     if (state == AppLifecycleState.inactive && _controller != null) {
//       _controller!.pause();
//     } else if (state == AppLifecycleState.resumed && _controller != null) {
//       _controller!.play();
//     }
//   }

//   void _initializeVideo() {
//     print("Initializing video...");
//     _controller = VideoPlayerController.asset('assets/images/page_1.mp4')
//       ..initialize().then((_) {
//         print("Video initialized successfully.");
//         setState(() {
//           _controller?.setLooping(true);
//           _controller?.play();
//         });
//       }).catchError((error) {
//         print("Error initializing video: $error");
//       });
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
//         print("Attempting to send file to: $uri");
//         final request = http.MultipartRequest('POST', uri);
//         request.files.add(
//           http.MultipartFile.fromBytes('file', fileBytes!,
//               filename: uploadedFileName),
//         );

//         final response = await request.send();
//         print("Response status code: ${response.statusCode}");
//         if (response.statusCode == 200) {
//           final responseData = await response.stream.bytesToString();
//           final data = json.decode(responseData);
//           print("Response data: $data");

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
//         print("No file selected.");
//       }
//     } catch (e) {
//       setState(() {
//         isLoading = false;
//       });
//       print("Error: $e");
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
//           'mode': widget.mode, // Pass the mode to the backend
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
//           _controller != null && _controller!.value.isInitialized
//               ? Opacity(
//                   opacity: 0.8,
//                   child: SizedBox.expand(
//                     child: FittedBox(
//                       fit: BoxFit.cover,
//                       child: SizedBox(
//                         width: _controller!.value.size.width,
//                         height: _controller!.value.size.height,
//                         child: VideoPlayer(_controller!),
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
//                           border: OutlineInputBorder(),
//                           filled: true,
//                           fillColor: Colors.white70,
//                         ),
//                       ),
//                       const SizedBox(height: 10),
//                       TextField(
//                         onChanged: (value) => yColumn = value,
//                         decoration: const InputDecoration(
//                           labelText: 'Y Column',
//                           border: OutlineInputBorder(),
//                           filled: true,
//                           fillColor: Colors.white70,
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



