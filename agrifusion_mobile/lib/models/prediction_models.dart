// Data models for API communication.

class PredictionRequest {
  final List<int> imageBytes;
  final String imageFileName;
  final double nitrogen;
  final double phosphorus;
  final double potassium;
  final double temperature;
  final double humidity;
  final double rainfall;
  final double ph;
  final double yieldLastSeason;
  final double fertilizerUsed;
  final String season;
  final String irrigation;
  final String previousCrop;
  final String region;

  PredictionRequest({
    required this.imageBytes,
    required this.imageFileName,
    required this.nitrogen,
    required this.phosphorus,
    required this.potassium,
    required this.temperature,
    required this.humidity,
    required this.rainfall,
    required this.ph,
    required this.yieldLastSeason,
    required this.fertilizerUsed,
    required this.season,
    required this.irrigation,
    required this.previousCrop,
    required this.region,
  });
}

class CropRecommendation {
  final String name;
  final int rank;
  final int stars;
  final String fertilizer;
  final String npk;

  CropRecommendation({
    required this.name,
    required this.rank,
    required this.stars,
    required this.fertilizer,
    required this.npk,
  });

  factory CropRecommendation.fromJson(Map<String, dynamic> json) {
    return CropRecommendation(
      name: json['name'] ?? 'Unknown',
      rank: json['rank'] ?? 0,
      stars: json['stars'] ?? 0,
      fertilizer: json['fertilizer'] ?? 'N/A',
      npk: json['npk'] ?? 'N/A',
    );
  }
}

class PredictionResult {
  final String soilType;
  final double confidence;
  final Map<String, double> allProbabilities;
  final String soilFertilizer;
  final String soilNpk;
  final List<CropRecommendation> recommendedCrops;
  final String season;
  final String region;

  PredictionResult({
    required this.soilType,
    required this.confidence,
    required this.allProbabilities,
    required this.soilFertilizer,
    required this.soilNpk,
    required this.recommendedCrops,
    required this.season,
    required this.region,
  });

  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    // Handle both API response formats (api.py and app.py)
    final soilType = json['soil_type'] ?? json['soil_name'] ?? 'Unknown';
    final confidence = (json['confidence'] ?? 0.0).toDouble();

    // Parse probabilities
    Map<String, double> probs = {};
    final rawProbs =
        json['all_probabilities'] ?? json['all_probs'] ?? {};
    if (rawProbs is Map) {
      rawProbs.forEach((key, value) {
        probs[key.toString()] = (value ?? 0.0).toDouble();
      });
    }

    // Parse soil fertilizer info
    String soilFert = 'N/A';
    String soilNpk = 'N/A';
    if (json['soil_fertilizer'] != null) {
      soilFert = json['soil_fertilizer'];
      soilNpk = json['soil_npk'] ?? 'N/A';
    } else if (json['soil_fert'] is Map) {
      soilFert = json['soil_fert']['fertilizer'] ?? 'N/A';
      soilNpk = json['soil_fert']['npk'] ?? 'N/A';
    }

    // Parse crop recommendations
    List<CropRecommendation> crops = [];
    final rawCrops =
        json['recommended_crops'] ?? json['crop_recs'] ?? [];
    if (rawCrops is List) {
      for (var c in rawCrops) {
        if (c is Map<String, dynamic>) {
          crops.add(CropRecommendation.fromJson(c));
        }
      }
    }

    return PredictionResult(
      soilType: soilType,
      confidence: confidence,
      allProbabilities: probs,
      soilFertilizer: soilFert,
      soilNpk: soilNpk,
      recommendedCrops: crops,
      season: json['season'] ?? '',
      region: json['region'] ?? '',
    );
  }
}

class HealthStatus {
  final bool isOk;
  final List<String> classes;
  final String? accuracy;

  HealthStatus({
    required this.isOk,
    required this.classes,
    this.accuracy,
  });

  factory HealthStatus.fromJson(Map<String, dynamic> json) {
    return HealthStatus(
      isOk: json['status'] == 'ok',
      classes: List<String>.from(json['classes'] ?? []),
      accuracy: json['accuracy'],
    );
  }
}
