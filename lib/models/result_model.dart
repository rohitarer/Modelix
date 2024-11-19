class ResultModel {
  final String bestModel;
  final double bestMSE;
  final double bestR2Score;
  final List<Map<String, dynamic>> otherModels;

  ResultModel({
    required this.bestModel,
    required this.bestMSE,
    required this.bestR2Score,
    required this.otherModels,
  });

  // Factory constructor to parse JSON
  factory ResultModel.fromJson(Map<String, dynamic> json) {
    return ResultModel(
      bestModel: json['best_model'],
      bestMSE: (json['mse'] as num).toDouble(),
      bestR2Score: (json['r2'] as num).toDouble(),
      otherModels: List<Map<String, dynamic>>.from(json['model_results']),
    );
  }
}
