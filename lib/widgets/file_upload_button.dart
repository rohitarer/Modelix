import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';

class FileUploadButton extends StatelessWidget {
  final void Function(String filePath) onFileSelected;

  const FileUploadButton({super.key, required this.onFileSelected});

  Future<void> _pickFile() async {
    try {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['csv'],
      );

      if (result != null) {
        onFileSelected(result.files.single.path!);
      }
    } catch (e) {
      // Handle errors or display a message
      debugPrint('Error picking file: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return ElevatedButton(
      onPressed: _pickFile,
      child: const Text('Upload CSV File'),
    );
  }
}
