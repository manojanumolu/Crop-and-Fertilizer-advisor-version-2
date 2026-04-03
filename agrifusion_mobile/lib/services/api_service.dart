import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:mime/mime.dart';
import '../models/prediction_models.dart';

/// Service class to communicate with the Hugging Face hosted AgriFusion API.
class ApiService {
  /// The base URL for the Hugging Face Space API.
  static const String baseUrl =
      'https://manoj-27-agrifusion-ai-api.hf.space';

  /// Timeout duration for API calls.
  static const Duration timeout = Duration(seconds: 120);

  /// Check if the API is healthy and models are loaded.
  static Future<HealthStatus> checkHealth() async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/health'))
          .timeout(timeout);

      if (response.statusCode == 200) {
        final json = jsonDecode(response.body);
        return HealthStatus.fromJson(json);
      } else {
        throw ApiException(
          'Server returned status ${response.statusCode}',
          statusCode: response.statusCode,
        );
      }
    } on http.ClientException catch (e) {
      throw ApiException(
        'Cannot connect to server. Please check your internet connection.',
        originalError: e,
      );
    } catch (e) {
      if (e is ApiException) rethrow;
      throw ApiException(
        'Connection failed: ${e.toString()}',
        originalError: e,
      );
    }
  }

  /// Validate if the uploaded image is a soil image.
  static Future<SoilValidationResult> validateSoilImage(
      Uint8List imageBytes, String fileName) async {
    try {
      final mimeType = lookupMimeType(fileName) ?? 'image/jpeg';
      final mimeParts = mimeType.split('/');

      final request =
          http.MultipartRequest('POST', Uri.parse('$baseUrl/predict'));

      request.files.add(http.MultipartFile.fromBytes(
        'image',
        imageBytes,
        filename: fileName,
        contentType: MediaType(mimeParts[0], mimeParts[1]),
      ));

      // Send with default values — MUST match api.py field names exactly
      request.fields['n'] = '90';
      request.fields['p'] = '42';
      request.fields['k'] = '43';
      request.fields['temp'] = '25';
      request.fields['hum'] = '80';
      request.fields['rain'] = '200';
      request.fields['ph'] = '6.5';
      request.fields['yld'] = '2500';
      request.fields['fert'] = '120';
      request.fields['season'] = 'Kharif';
      request.fields['irrig'] = 'Canal';
      request.fields['prev'] = 'Wheat';
      request.fields['region'] = 'South';

      final streamed = await request.send().timeout(timeout);
      final response = await http.Response.fromStream(streamed);

      if (response.statusCode == 200) {
        return SoilValidationResult(isValid: true);
      } else if (response.statusCode == 400) {
        final json = jsonDecode(response.body);
        final error = json['error'] ?? '';
        if (error.toString().toLowerCase().contains('soil')) {
          return SoilValidationResult(
            isValid: false,
            message: error.toString(),
          );
        }
        return SoilValidationResult(isValid: true);
      } else {
        throw ApiException(
          'Validation failed with status ${response.statusCode}',
          statusCode: response.statusCode,
        );
      }
    } on http.ClientException catch (e) {
      throw ApiException(
        'Cannot connect to server for validation.',
        originalError: e,
      );
    } catch (e) {
      if (e is ApiException) rethrow;
      throw ApiException(
        'Image validation error: ${e.toString()}',
        originalError: e,
      );
    }
  }

  /// Send prediction request with both image and numerical data.
  static Future<PredictionResult> predict(
      PredictionRequest request) async {
    try {
      final mimeType =
          lookupMimeType(request.imageFileName) ?? 'image/jpeg';
      final mimeParts = mimeType.split('/');

      final multipart = http.MultipartRequest(
          'POST', Uri.parse('$baseUrl/predict'));

      // Add image file
      multipart.files.add(http.MultipartFile.fromBytes(
        'image',
        request.imageBytes,
        filename: request.imageFileName,
        contentType: MediaType(mimeParts[0], mimeParts[1]),
      ));

      // ═══════════════════════════════════════════════════════
      // CRITICAL: Field names MUST match api.py exactly!
      // api.py uses: n, p, k, temp, hum, rain, ph, yld, fert
      //              season, irrig, prev, region
      // ═══════════════════════════════════════════════════════
      multipart.fields['n'] = request.nitrogen.toString();
      multipart.fields['p'] = request.phosphorus.toString();
      multipart.fields['k'] = request.potassium.toString();
      multipart.fields['temp'] = request.temperature.toString();
      multipart.fields['hum'] = request.humidity.toString();
      multipart.fields['rain'] = request.rainfall.toString();
      multipart.fields['ph'] = request.ph.toString();
      multipart.fields['yld'] = request.yieldLastSeason.toString();
      multipart.fields['fert'] = request.fertilizerUsed.toString();
      multipart.fields['season'] = request.season;
      multipart.fields['irrig'] = request.irrigation;
      multipart.fields['prev'] = request.previousCrop;
      multipart.fields['region'] = request.region;

      final streamed =
          await multipart.send().timeout(timeout);
      final response = await http.Response.fromStream(streamed);

      if (response.statusCode == 200) {
        final json = jsonDecode(response.body);
        return PredictionResult.fromJson(json);
      } else if (response.statusCode == 400) {
        final json = jsonDecode(response.body);
        throw ApiException(
          json['error'] ?? 'Invalid request',
          statusCode: 400,
        );
      } else {
        throw ApiException(
          'Prediction failed (status ${response.statusCode})',
          statusCode: response.statusCode,
        );
      }
    } on http.ClientException catch (e) {
      throw ApiException(
        'No internet connection. Please check your network and try again.',
        originalError: e,
      );
    } catch (e) {
      if (e is ApiException) rethrow;
      throw ApiException(
        'Prediction error: ${e.toString()}',
        originalError: e,
      );
    }
  }
}

/// Result of soil image validation.
class SoilValidationResult {
  final bool isValid;
  final String? message;

  SoilValidationResult({required this.isValid, this.message});
}

/// Custom exception for API errors.
class ApiException implements Exception {
  final String message;
  final int? statusCode;
  final Object? originalError;

  ApiException(this.message, {this.statusCode, this.originalError});

  @override
  String toString() => 'ApiException: $message';
}
