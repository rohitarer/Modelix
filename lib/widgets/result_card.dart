import 'package:flutter/material.dart';
import '../models/result_model.dart';

class ResultCard extends StatelessWidget {
  final ResultModel result;

  const ResultCard({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Best Model: ${result.bestModel}',
              style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            Text(
              'Best MSE: ${result.bestMSE.toStringAsFixed(2)}',
              style: const TextStyle(fontSize: 16),
            ),
            Text(
              'Best R2 Score: ${result.bestR2Score.toStringAsFixed(2)}',
              style: const TextStyle(fontSize: 16),
            ),
            const Divider(height: 24),
            const Text(
              'Other Models:',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
            ),
            ...result.otherModels.map(
              (model) => Padding(
                padding: const EdgeInsets.symmetric(vertical: 4.0),
                child: ListTile(
                  contentPadding: EdgeInsets.zero,
                  title: Text(model['Model']),
                  subtitle: Text(
                    'MSE: ${model['MSE']}, R2: ${model['R2 Score']}',
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
