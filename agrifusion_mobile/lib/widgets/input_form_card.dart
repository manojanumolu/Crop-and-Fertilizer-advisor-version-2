import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:animate_do/animate_do.dart';
import '../theme/app_theme.dart';
import '../data/india_geo_data.dart';
import '../services/weather_service.dart';

// Ideal ranges for tooltip display
const Map<String, String> idealRanges = {
  'nitrogen': 'Ideal: 60–140 mg/kg\nLow < 60 · High > 140',
  'phosphorus': 'Ideal: 25–80 mg/kg\nLow < 25 · High > 80',
  'potassium': 'Ideal: 35–120 mg/kg\nLow < 35 · High > 120',
  'ph': 'Ideal: pH 6.0–7.5\nAcidic < 6 · Alkaline > 7.5',
  'temperature': 'Ideal: 15–35 °C\nCrop-dependent range',
  'humidity': 'Ideal: 50–85%\nLow < 50 · High > 85',
  'rainfall': 'Ideal: 500–2500 mm/year\nVaries by region & crop',
  'yield': 'Typical: 1000–5000 t/ha\nUse tonnes per hectare',
  'fertilizer': 'Typical: 50–300 kg/ha\nSum all NPK applied last season',
};

class InputFormCard extends StatefulWidget {
  final double nitrogen, phosphorus, potassium;
  final double temperature, humidity, rainfall;
  final double ph, yieldLastSeason, fertilizerUsed;
  final String season, irrigation, previousCrop, region;
  final bool isPredicting;

  final ValueChanged<double> onNitrogenChanged;
  final ValueChanged<double> onPhosphorusChanged;
  final ValueChanged<double> onPotassiumChanged;
  final ValueChanged<double> onTemperatureChanged;
  final ValueChanged<double> onHumidityChanged;
  final ValueChanged<double> onRainfallChanged;
  final ValueChanged<double> onPhChanged;
  final ValueChanged<double> onYieldChanged;
  final ValueChanged<double> onFertilizerChanged;
  final ValueChanged<String> onSeasonChanged;
  final ValueChanged<String> onIrrigationChanged;
  final ValueChanged<String> onPreviousCropChanged;
  final ValueChanged<String> onRegionChanged;
  final VoidCallback onSubmit;

  const InputFormCard({
    super.key,
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
    required this.isPredicting,
    required this.onNitrogenChanged,
    required this.onPhosphorusChanged,
    required this.onPotassiumChanged,
    required this.onTemperatureChanged,
    required this.onHumidityChanged,
    required this.onRainfallChanged,
    required this.onPhChanged,
    required this.onYieldChanged,
    required this.onFertilizerChanged,
    required this.onSeasonChanged,
    required this.onIrrigationChanged,
    required this.onPreviousCropChanged,
    required this.onRegionChanged,
    required this.onSubmit,
  });

  @override
  State<InputFormCard> createState() => _InputFormCardState();
}

class _InputFormCardState extends State<InputFormCard> {
  String? _selectedState;
  String? _selectedDistrict;
  final TextEditingController _villageController = TextEditingController();
  bool _isFetchingWeather = false;
  String? _weatherNote;

  @override
  void dispose() {
    _villageController.dispose();
    super.dispose();
  }

  List<String> get _districts {
    if (_selectedState == null) return [];
    return indiaStatesDistricts[_selectedState!] ?? [];
  }

  Future<void> _fetchWeatherData() async {
    if (_selectedState == null || _selectedDistrict == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Please select State and District first', style: GoogleFonts.workSans(fontSize: 13)),
          backgroundColor: AppColors.danger,
          behavior: SnackBarBehavior.floating,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
        ),
      );
      return;
    }

    setState(() { _isFetchingWeather = true; _weatherNote = null; });

    try {
      final result = await WeatherService.fetchClimateData(
        state: _selectedState!,
        district: _selectedDistrict!,
        village: _villageController.text,
      );
      widget.onTemperatureChanged(result.temperature);
      widget.onHumidityChanged(result.humidity);
      widget.onRainfallChanged(result.rainfall);
      setState(() {
        _isFetchingWeather = false;
        _weatherNote = '${result.locationLabel} • ${result.note}';
      });
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Row(
              children: [
                const Icon(Icons.check_circle_outline, color: Colors.white, size: 18),
                const SizedBox(width: 8),
                Expanded(child: Text('Climate data loaded: ${result.temperature}°C, ${result.humidity}%, ${result.rainfall}mm', style: GoogleFonts.workSans(fontSize: 12))),
              ],
            ),
            backgroundColor: AppColors.primary,
            behavior: SnackBarBehavior.floating,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
          ),
        );
      }
    } catch (e) {
      setState(() {
        _isFetchingWeather = false;
        _weatherNote = 'Error: ${e.toString().replaceAll('Exception: ', '')}';
      });
    }
  }

  // ═══════════════════════════════════════════════════════
  // Searchable Location Picker — opens a bottom sheet
  // ═══════════════════════════════════════════════════════
  Future<String?> _showSearchableBottomSheet(String title, List<String> items) async {
    return showModalBottomSheet<String>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (ctx) => _SearchableListSheet(title: title, items: items),
    );
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 32),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // ────────────────────────────────────────
          // Chemical Profile Section
          // ────────────────────────────────────────
          FadeInDown(
            duration: const Duration(milliseconds: 400),
            child: _buildSectionCard(
              icon: Icons.science_rounded,
              title: 'Chemical Profile',
              subtitle: 'Soil NPK values (mg/kg)',
              cardColor: const Color(0xFFF0FFF4),
              child: Column(
                children: [
                  _buildNumberBox(label: 'NITROGEN (N)', unit: 'mg/kg', value: widget.nitrogen, step: 1, min: 0, max: 300, tooltipKey: 'nitrogen', onChanged: widget.onNitrogenChanged),
                  const SizedBox(height: 14),
                  _buildNumberBox(label: 'PHOSPHORUS (P)', unit: 'mg/kg', value: widget.phosphorus, step: 1, min: 0, max: 200, tooltipKey: 'phosphorus', onChanged: widget.onPhosphorusChanged),
                  const SizedBox(height: 14),
                  _buildNumberBox(label: 'POTASSIUM (K)', unit: 'mg/kg', value: widget.potassium, step: 1, min: 0, max: 200, tooltipKey: 'potassium', onChanged: widget.onPotassiumChanged),
                  const SizedBox(height: 14),
                  _buildNumberBox(label: 'SOIL PH', unit: 'pH', value: widget.ph, step: 0.1, min: 3.0, max: 10.0, decimals: 1, tooltipKey: 'ph', onChanged: widget.onPhChanged),
                  const SizedBox(height: 10),
                  _buildInfoBanner('Ideal bands → N: 60-140, P: 25-80, K: 35-120 mg/kg, pH: 6.0-7.5'),
                ],
              ),
            ),
          ),

          const SizedBox(height: 16),

          // ────────────────────────────────────────
          // Auto-Fill Climate Data Section
          // ────────────────────────────────────────
          FadeInUp(
            delay: const Duration(milliseconds: 100),
            duration: const Duration(milliseconds: 400),
            child: _buildSectionCard(
              icon: Icons.cloud_sync_rounded,
              title: 'Auto-Fill Climate Data',
              subtitle: 'Fetch real weather by location',
              cardColor: const Color(0xFFF0F8FF),
              child: Column(
                children: [
                  // State — searchable
                  _buildSearchablePicker(
                    label: '📍 SELECT YOUR STATE',
                    tooltip: 'Tap to search and select your state',
                    value: _selectedState,
                    hint: '-- Tap to search State --',
                    onTap: () async {
                      final states = indiaStatesDistricts.keys.toList()..sort();
                      final picked = await _showSearchableBottomSheet('Select State', states);
                      if (picked != null) {
                        setState(() {
                          _selectedState = picked;
                          _selectedDistrict = null;
                          _weatherNote = null;
                        });
                      }
                    },
                  ),
                  const SizedBox(height: 12),
                  // District — searchable
                  _buildSearchablePicker(
                    label: '🏛 SELECT YOUR DISTRICT',
                    tooltip: 'Tap to search and select your district',
                    value: _selectedDistrict,
                    hint: _selectedState == null ? '-- Select State First --' : '-- Tap to search District --',
                    onTap: _selectedState == null ? null : () async {
                      final picked = await _showSearchableBottomSheet('Select District in $_selectedState', _districts);
                      if (picked != null) {
                        setState(() {
                          _selectedDistrict = picked;
                          _weatherNote = null;
                        });
                      }
                    },
                  ),
                  const SizedBox(height: 12),
                  // Village text field
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _buildFieldLabel('🏘 VILLAGE / TOWN (OPTIONAL)', 'Optional village name for precise data'),
                      const SizedBox(height: 6),
                      Container(
                        decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(10), border: Border.all(color: AppColors.surfaceDim)),
                        child: TextField(
                          controller: _villageController,
                          style: GoogleFonts.workSans(fontSize: 14, fontWeight: FontWeight.w600, color: AppColors.textPrimary),
                          decoration: InputDecoration(
                            hintText: 'e.g. Ramtek',
                            hintStyle: GoogleFonts.workSans(fontSize: 13, color: AppColors.textMuted),
                            contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
                            border: OutlineInputBorder(borderRadius: BorderRadius.circular(10), borderSide: BorderSide.none),
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  // Fetch button
                  SizedBox(
                    width: double.infinity,
                    height: 48,
                    child: ElevatedButton.icon(
                      onPressed: _isFetchingWeather ? null : _fetchWeatherData,
                      icon: _isFetchingWeather
                          ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                          : const Icon(Icons.cloud_download_rounded, size: 18),
                      label: Text(
                        _isFetchingWeather ? 'Fetching...' : '🌍 Fetch Local Climate Data',
                        style: GoogleFonts.manrope(fontSize: 13, fontWeight: FontWeight.w700),
                      ),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: AppColors.primary,
                        foregroundColor: Colors.white,
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                        elevation: 0,
                      ),
                    ),
                  ),

                  // Weather note
                  if (_weatherNote != null) ...[
                    const SizedBox(height: 12),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                      decoration: BoxDecoration(
                        color: _weatherNote!.startsWith('Error') ? AppColors.dangerSurface : AppColors.primarySurfaceLight,
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Row(
                        children: [
                          Icon(
                            _weatherNote!.startsWith('Error') ? Icons.warning_amber_rounded : Icons.location_on_rounded,
                            size: 14,
                            color: _weatherNote!.startsWith('Error') ? AppColors.danger : AppColors.primary,
                          ),
                          const SizedBox(width: 6),
                          Expanded(child: Text(_weatherNote!, style: GoogleFonts.workSans(fontSize: 11, color: _weatherNote!.startsWith('Error') ? AppColors.danger : AppColors.primary))),
                        ],
                      ),
                    ),
                  ],

                  const SizedBox(height: 16),

                  // Climate inputs — single column
                  _buildNumberBox(label: 'TEMPERATURE', unit: '°C', value: widget.temperature, step: 0.5, min: 0, max: 60, decimals: 1, tooltipKey: 'temperature', onChanged: widget.onTemperatureChanged),
                  const SizedBox(height: 14),
                  _buildNumberBox(label: 'HUMIDITY', unit: '%', value: widget.humidity, step: 0.5, min: 0, max: 100, decimals: 1, tooltipKey: 'humidity', onChanged: widget.onHumidityChanged),
                  const SizedBox(height: 14),
                  _buildNumberBox(label: 'ANNUAL RAINFALL', unit: 'mm/yr', value: widget.rainfall, step: 10, min: 0, max: 5000, decimals: 1, tooltipKey: 'rainfall', onChanged: widget.onRainfallChanged),
                ],
              ),
            ),
          ),

          const SizedBox(height: 16),

          // ────────────────────────────────────────
          // Farm History Section
          // ────────────────────────────────────────
          FadeInUp(
            delay: const Duration(milliseconds: 200),
            duration: const Duration(milliseconds: 400),
            child: _buildSectionCard(
              icon: Icons.bar_chart_rounded,
              title: 'Farm History',
              subtitle: 'Previous season metrics',
              cardColor: const Color(0xFFFFFBF0),
              child: Column(
                children: [
                  _buildNumberBox(label: 'YIELD LAST SEASON', unit: 't/ha', value: widget.yieldLastSeason, step: 100, min: 0, max: 20000, tooltipKey: 'yield', onChanged: widget.onYieldChanged),
                  const SizedBox(height: 14),
                  _buildNumberBox(label: 'FERTILIZER USED', unit: 'kg/ha', value: widget.fertilizerUsed, step: 10, min: 0, max: 1000, tooltipKey: 'fertilizer', onChanged: widget.onFertilizerChanged),
                ],
              ),
            ),
          ),

          const SizedBox(height: 16),

          // ────────────────────────────────────────
          // Farm Details Section
          // ────────────────────────────────────────
          FadeInUp(
            delay: const Duration(milliseconds: 300),
            duration: const Duration(milliseconds: 400),
            child: _buildSectionCard(
              icon: Icons.agriculture_rounded,
              title: 'Farm Details',
              subtitle: 'Season, irrigation & region',
              cardColor: const Color(0xFFF5F0FF),
              child: Column(
                children: [
                  _buildDropdownField(label: 'CURRENT SEASON', tooltip: 'Kharif=Jun-Oct, Rabi=Nov-Mar, Zaid=Mar-Jun', value: widget.season, items: const ['Kharif', 'Rabi', 'Zaid'], labels: const ['Kharif (Monsoon)', 'Rabi (Winter)', 'Zaid (Summer)'], onChanged: widget.onSeasonChanged),
                  const SizedBox(height: 14),
                  _buildDropdownField(label: 'IRRIGATION SYSTEM', tooltip: 'Type of water supply to your farm', value: widget.irrigation, items: const ['Canal', 'Drip', 'Rainfed', 'Sprinkler'], onChanged: widget.onIrrigationChanged),
                  const SizedBox(height: 14),
                  _buildDropdownField(label: 'PREVIOUS CROP', tooltip: 'Crop grown last season', value: widget.previousCrop, items: const ['Cotton', 'Maize', 'Potato', 'Rice', 'Sugarcane', 'Tomato', 'Wheat'], onChanged: widget.onPreviousCropChanged),
                  const SizedBox(height: 14),
                  _buildDropdownField(label: 'GEOGRAPHIC REGION', tooltip: 'Broad geographic zone', value: widget.region, items: const ['Central', 'East', 'North', 'South', 'West'], onChanged: widget.onRegionChanged),
                ],
              ),
            ),
          ),

          const SizedBox(height: 24),

          // Action buttons
          FadeInUp(
            delay: const Duration(milliseconds: 400),
            duration: const Duration(milliseconds: 400),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                ElevatedButton(
                  onPressed: widget.isPredicting ? null : widget.onSubmit,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppColors.primary,
                    foregroundColor: Colors.white,
                    disabledBackgroundColor: AppColors.primary.withValues(alpha: 0.6),
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
                    elevation: 2,
                  ),
                  child: widget.isPredicting
                      ? Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2.5, color: Colors.white)),
                            const SizedBox(width: 12),
                            Text('Analyzing Soil...', style: GoogleFonts.manrope(fontSize: 15, fontWeight: FontWeight.w700)),
                          ],
                        )
                      : Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            const Icon(Icons.auto_awesome_rounded, size: 20),
                            const SizedBox(width: 8),
                            Text('Analyze & Predict', style: GoogleFonts.manrope(fontSize: 15, fontWeight: FontWeight.w700)),
                          ],
                        ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // ═══════════════════════════════════════════════════════
  // Helper Widgets
  // ═══════════════════════════════════════════════════════

  Widget _buildFieldLabel(String label, String tooltipText) {
    return Row(
      children: [
        Flexible(
          child: Text(label, style: GoogleFonts.manrope(fontSize: 10, fontWeight: FontWeight.w700, color: AppColors.textSecondary, letterSpacing: 0.5), overflow: TextOverflow.ellipsis),
        ),
        const SizedBox(width: 4),
        Tooltip(
          message: tooltipText,
          triggerMode: TooltipTriggerMode.tap,
          preferBelow: false,
          decoration: BoxDecoration(color: AppColors.primaryDark, borderRadius: BorderRadius.circular(8)),
          textStyle: GoogleFonts.workSans(fontSize: 12, color: Colors.white),
          child: Icon(Icons.help_outline_rounded, size: 14, color: AppColors.textMuted.withValues(alpha: 0.6)),
        ),
      ],
    );
  }

  Widget _buildInfoBanner(String text) {
    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(color: AppColors.infoSurface, borderRadius: BorderRadius.circular(8)),
      child: Row(
        children: [
          const Icon(Icons.info_outline, size: 14, color: AppColors.info),
          const SizedBox(width: 8),
          Expanded(child: Text(text, style: GoogleFonts.workSans(fontSize: 11, color: AppColors.textMuted))),
        ],
      ),
    );
  }

  Widget _buildSectionCard({
    required IconData icon,
    required String title,
    required String subtitle,
    required Widget child,
    Color? cardColor,
  }) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: cardColor ?? AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.surfaceDim.withValues(alpha: 0.5), width: 0.5),
        boxShadow: [
          BoxShadow(color: Colors.black.withValues(alpha: 0.04), blurRadius: 12, offset: const Offset(0, 4)),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(color: AppColors.primarySurfaceLight, borderRadius: BorderRadius.circular(10)),
                child: Icon(icon, color: AppColors.primary, size: 20),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(title, style: GoogleFonts.manrope(fontSize: 16, fontWeight: FontWeight.w700, color: AppColors.textPrimary)),
                    Text(subtitle, style: GoogleFonts.workSans(fontSize: 12, color: AppColors.textMuted)),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          child,
        ],
      ),
    );
  }

  Widget _buildNumberBox({
    required String label,
    required String unit,
    required double value,
    required double step,
    required double min,
    required double max,
    int decimals = 0,
    required String tooltipKey,
    required ValueChanged<double> onChanged,
  }) {
    final displayValue = decimals > 0 ? value.toStringAsFixed(decimals) : value.toStringAsFixed(0);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _buildFieldLabel('$label ($unit)', idealRanges[tooltipKey] ?? ''),
        const SizedBox(height: 6),
        Container(
          height: 50,
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: AppColors.primary.withValues(alpha: 0.15), width: 1),
          ),
          child: Row(
            children: [
              _buildStepButton(icon: Icons.remove, isLeft: true, onTap: () {
                final newVal = (value - step).clamp(min, max);
                onChanged(double.parse(newVal.toStringAsFixed(decimals)));
              }),
              Expanded(
                child: TextFormField(
                  key: ValueKey('$tooltipKey-$displayValue'),
                  initialValue: displayValue,
                  textAlign: TextAlign.center,
                  keyboardType: const TextInputType.numberWithOptions(decimal: true),
                  inputFormatters: [FilteringTextInputFormatter.allow(RegExp(r'^\d*\.?\d{0,2}'))],
                  style: GoogleFonts.manrope(fontSize: 16, fontWeight: FontWeight.w700, color: AppColors.textPrimary),
                  decoration: const InputDecoration(contentPadding: EdgeInsets.symmetric(horizontal: 8, vertical: 12), border: InputBorder.none),
                  onChanged: (text) {
                    final parsed = double.tryParse(text);
                    if (parsed != null) onChanged(parsed.clamp(min, max));
                  },
                ),
              ),
              Padding(
                padding: const EdgeInsets.only(right: 4),
                child: Text(unit, style: GoogleFonts.workSans(fontSize: 9, color: AppColors.textMuted, fontWeight: FontWeight.w500)),
              ),
              _buildStepButton(icon: Icons.add, isLeft: false, onTap: () {
                final newVal = (value + step).clamp(min, max);
                onChanged(double.parse(newVal.toStringAsFixed(decimals)));
              }),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildStepButton({required IconData icon, required bool isLeft, required VoidCallback onTap}) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.horizontal(
          left: isLeft ? const Radius.circular(12) : Radius.zero,
          right: isLeft ? Radius.zero : const Radius.circular(12),
        ),
        child: Container(
          width: 46,
          height: 50,
          decoration: BoxDecoration(
            color: AppColors.primary.withValues(alpha: 0.08),
            border: Border(
              right: isLeft ? BorderSide(color: AppColors.primary.withValues(alpha: 0.12), width: 1) : BorderSide.none,
              left: !isLeft ? BorderSide(color: AppColors.primary.withValues(alpha: 0.12), width: 1) : BorderSide.none,
            ),
          ),
          child: Center(child: Icon(icon, size: 20, color: AppColors.primary)),
        ),
      ),
    );
  }

  // Searchable picker — tappable field that opens a search bottom sheet
  Widget _buildSearchablePicker({
    required String label,
    required String tooltip,
    required String? value,
    required String hint,
    required VoidCallback? onTap,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _buildFieldLabel(label, tooltip),
        const SizedBox(height: 6),
        GestureDetector(
          onTap: onTap,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 14),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: AppColors.surfaceDim),
            ),
            child: Row(
              children: [
                Expanded(
                  child: Text(
                    value ?? hint,
                    style: GoogleFonts.workSans(
                      fontSize: 13,
                      fontWeight: value != null ? FontWeight.w600 : FontWeight.w400,
                      color: value != null ? AppColors.textPrimary : AppColors.textMuted,
                    ),
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
                const Icon(Icons.search_rounded, size: 18, color: AppColors.textMuted),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildDropdownField({
    required String label,
    required String tooltip,
    required String value,
    required List<String> items,
    List<String>? labels,
    required ValueChanged<String> onChanged,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _buildFieldLabel(label, tooltip),
        const SizedBox(height: 6),
        Container(
          decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(10), border: Border.all(color: AppColors.surfaceDim)),
          child: DropdownButtonFormField<String>(
            value: value,
            decoration: InputDecoration(
              contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
              border: OutlineInputBorder(borderRadius: BorderRadius.circular(10), borderSide: BorderSide.none),
            ),
            dropdownColor: AppColors.surface,
            borderRadius: BorderRadius.circular(10),
            isExpanded: true,
            style: GoogleFonts.workSans(fontSize: 13, fontWeight: FontWeight.w600, color: AppColors.textPrimary),
            items: List.generate(items.length, (i) => DropdownMenuItem<String>(
              value: items[i],
              child: Text(labels != null ? labels[i] : items[i], style: GoogleFonts.workSans(fontSize: 13, color: AppColors.textPrimary)),
            )),
            onChanged: (v) { if (v != null) onChanged(v); },
          ),
        ),
      ],
    );
  }
}

// ═══════════════════════════════════════════════════════
// Searchable Bottom Sheet Widget
// ═══════════════════════════════════════════════════════
class _SearchableListSheet extends StatefulWidget {
  final String title;
  final List<String> items;

  const _SearchableListSheet({required this.title, required this.items});

  @override
  State<_SearchableListSheet> createState() => _SearchableListSheetState();
}

class _SearchableListSheetState extends State<_SearchableListSheet> {
  final TextEditingController _searchCtrl = TextEditingController();
  List<String> _filtered = [];

  @override
  void initState() {
    super.initState();
    _filtered = List.from(widget.items);
  }

  @override
  void dispose() {
    _searchCtrl.dispose();
    super.dispose();
  }

  void _onSearch(String query) {
    setState(() {
      if (query.isEmpty) {
        _filtered = List.from(widget.items);
      } else {
        _filtered = widget.items.where((item) => item.toLowerCase().contains(query.toLowerCase())).toList();
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final height = MediaQuery.of(context).size.height * 0.65;
    return Container(
      height: height,
      decoration: const BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      child: Column(
        children: [
          // Handle
          Center(
            child: Container(
              margin: const EdgeInsets.only(top: 10),
              width: 40,
              height: 4,
              decoration: BoxDecoration(color: AppColors.surfaceDim, borderRadius: BorderRadius.circular(2)),
            ),
          ),
          // Title
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 14, 20, 0),
            child: Text(widget.title, style: GoogleFonts.manrope(fontSize: 16, fontWeight: FontWeight.w700, color: AppColors.textPrimary)),
          ),
          // Search box
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 8),
            child: Container(
              decoration: BoxDecoration(
                color: AppColors.surfaceVariant,
                borderRadius: BorderRadius.circular(12),
              ),
              child: TextField(
                controller: _searchCtrl,
                autofocus: true,
                style: GoogleFonts.workSans(fontSize: 14, color: AppColors.textPrimary),
                decoration: InputDecoration(
                  hintText: 'Type to search...',
                  hintStyle: GoogleFonts.workSans(fontSize: 14, color: AppColors.textMuted),
                  prefixIcon: const Icon(Icons.search_rounded, size: 20, color: AppColors.textMuted),
                  contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                  border: InputBorder.none,
                ),
                onChanged: _onSearch,
              ),
            ),
          ),
          const Divider(height: 1),
          // Results list
          Expanded(
            child: _filtered.isEmpty
                ? Center(child: Text('No results found', style: GoogleFonts.workSans(fontSize: 14, color: AppColors.textMuted)))
                : ListView.builder(
                    itemCount: _filtered.length,
                    padding: const EdgeInsets.symmetric(vertical: 4),
                    itemBuilder: (ctx, i) {
                      return ListTile(
                        dense: true,
                        title: Text(_filtered[i], style: GoogleFonts.workSans(fontSize: 14, fontWeight: FontWeight.w500, color: AppColors.textPrimary)),
                        trailing: const Icon(Icons.chevron_right_rounded, size: 18, color: AppColors.textMuted),
                        onTap: () => Navigator.pop(context, _filtered[i]),
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }
}
