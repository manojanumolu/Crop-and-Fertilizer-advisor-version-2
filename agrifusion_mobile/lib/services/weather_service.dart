import 'dart:convert';
import 'package:http/http.dart' as http;
import '../data/india_geo_data.dart';

class ClimateResult {
  final double temperature;
  final double humidity;
  final double rainfall;
  final String locationLabel;
  final String note;

  ClimateResult({
    required this.temperature,
    required this.humidity,
    required this.rainfall,
    required this.locationLabel,
    required this.note,
  });
}

class WeatherService {
  /// Fetches 10-year averaged climate data using Open-Meteo Archive API.
  /// Mirrors the get_climate_data() function from streamlit_app.py.
  static Future<ClimateResult> fetchClimateData({
    required String state,
    required String district,
    String? village,
  }) async {
    double lat;
    double lon;
    String locationLabel;
    String note;

    // Step 1: Get district coordinates
    final coordKey = '$state|$district';
    final coords = districtCoords[coordKey];
    if (coords != null) {
      lat = coords[0];
      lon = coords[1];
    } else {
      // Fallback: try geocoding the district name
      try {
        final geoUrl = Uri.parse(
          'https://geocoding-api.open-meteo.com/v1/search'
          '?name=${Uri.encodeComponent(district)}'
          '&count=5&language=en&format=json',
        );
        final geoResp = await http.get(geoUrl).timeout(
          const Duration(seconds: 8),
        );
        final geoData = json.decode(geoResp.body);
        final results = (geoData['results'] as List? ?? []);
        final indiaResults = results
            .where((r) =>
                (r['country_code'] ?? '').toString().toUpperCase() == 'IN')
            .toList();

        if (indiaResults.isNotEmpty) {
          lat = (indiaResults[0]['latitude'] as num).toDouble();
          lon = (indiaResults[0]['longitude'] as num).toDouble();
        } else {
          throw Exception('Could not find coordinates for $district, $state');
        }
      } catch (e) {
        throw Exception('Coordinates not found for $district, $state');
      }
    }

    // Step 2: Refine with village geocoding (like the website)
    if (village != null && village.trim().isNotEmpty) {
      try {
        final geoUrl = Uri.parse(
          'https://geocoding-api.open-meteo.com/v1/search'
          '?name=${Uri.encodeComponent(village.trim())}'
          '&count=5&language=en&format=json',
        );
        final geoResp = await http.get(geoUrl).timeout(
          const Duration(seconds: 5),
        );
        final geoData = json.decode(geoResp.body);
        final results = (geoData['results'] as List? ?? []);
        final indiaResults = results
            .where((r) =>
                (r['country_code'] ?? '').toString().toUpperCase() == 'IN')
            .toList();
        final stateResults = indiaResults
            .where((r) =>
                (r['admin1'] ?? '').toString().toLowerCase().contains(
                    state.toLowerCase()))
            .toList();

        if (stateResults.isNotEmpty) {
          lat = (stateResults[0]['latitude'] as num).toDouble();
          lon = (stateResults[0]['longitude'] as num).toDouble();
          locationLabel = '$village, $district, $state';
          note = 'Village location found ✓';
        } else {
          locationLabel = '$village, $district, $state';
          note = 'Using $district district coordinates';
        }
      } catch (_) {
        locationLabel = '$village, $district, $state';
        note = 'Using $district district coordinates';
      }
    } else {
      locationLabel = '$district, $state';
      note = 'District coordinates used';
    }

    // Step 3: Fetch 10-year climate data from Open-Meteo Archive
    final climateUrl = Uri.parse(
      'https://archive-api.open-meteo.com/v1/archive'
      '?latitude=$lat&longitude=$lon'
      '&start_date=2014-01-01&end_date=2023-12-31'
      '&daily=temperature_2m_mean,precipitation_sum'
      '&hourly=relative_humidity_2m'
      '&timezone=Asia%2FKolkata',
    );

    final climateResp = await http.get(climateUrl).timeout(
      const Duration(seconds: 60),
    );

    if (climateResp.statusCode != 200) {
      throw Exception('Weather API error: ${climateResp.statusCode}');
    }

    final climateData = json.decode(climateResp.body);
    final daily = climateData['daily'] as Map<String, dynamic>? ?? {};
    final hourly = climateData['hourly'] as Map<String, dynamic>? ?? {};

    final tempsAll = (daily['temperature_2m_mean'] as List? ?? []);
    final rainsAll = (daily['precipitation_sum'] as List? ?? []);
    final humsAll = (hourly['relative_humidity_2m'] as List? ?? []);

    // Temperature: mean of daily values
    final temps = tempsAll.where((t) => t != null).map((t) => (t as num).toDouble()).toList();
    final avgTemp = temps.isNotEmpty
        ? double.parse((temps.reduce((a, b) => a + b) / temps.length).toStringAsFixed(1))
        : 25.0;

    // Rainfall: annual average
    final nDays = tempsAll.isNotEmpty ? tempsAll.length : 3652;
    final rains = rainsAll.map((r) => r != null ? (r as num).toDouble() : 0.0).toList();
    final totalRain = rains.isNotEmpty ? rains.reduce((a, b) => a + b) : 0.0;
    final annualRain = double.parse((totalRain / (nDays / 365.0)).toStringAsFixed(1));

    // Humidity: mean of hourly ERA5 values
    final hums = humsAll.where((h) => h != null).map((h) => (h as num).toDouble()).toList();
    final avgHum = hums.isNotEmpty
        ? double.parse((hums.reduce((a, b) => a + b) / hums.length).toStringAsFixed(1))
        : 60.0;

    return ClimateResult(
      temperature: avgTemp,
      humidity: avgHum,
      rainfall: annualRain,
      locationLabel: locationLabel,
      note: note,
    );
  }
}
