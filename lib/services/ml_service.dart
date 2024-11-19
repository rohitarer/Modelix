import 'dart:io';
import 'dart:convert';

class MLService {
  /// Runs the ML model executable with the given CSV file path
  Future<Map<String, dynamic>> runMLModel(String csvFilePath) async {
    try {
      final result = await Process.run(
        'assets/ml_model', // Path to the executable
        [csvFilePath], // Arguments
      );

      if (result.exitCode == 0) {
        // Parse the output (assuming JSON format)
        final output = result.stdout.toString();
        return output.isNotEmpty ? jsonDecode(output) : {};
      } else {
        throw Exception(result.stderr.toString());
      }
    } catch (e) {
      throw Exception('Failed to execute ML model: $e');
    }
  }
}
