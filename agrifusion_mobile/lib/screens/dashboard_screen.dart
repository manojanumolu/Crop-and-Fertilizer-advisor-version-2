import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';
import '../theme/app_theme.dart';
import '../services/api_service.dart';
import '../models/prediction_models.dart';
import '../widgets/step_indicator.dart';
import '../widgets/soil_upload_card.dart';
import '../widgets/input_form_card.dart';
import '../widgets/result_view.dart';

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen>
    with TickerProviderStateMixin {
  // Step control
  int _currentStep = 0;
  final PageController _pageController = PageController();

  // Image state
  Uint8List? _imageBytes;
  String _imageFileName = '';
  bool _isValidatingSoil = false;
  bool _soilValidated = false;
  String? _validationError;

  // Form values
  double _nitrogen = 90;
  double _phosphorus = 42;
  double _potassium = 43;
  double _temperature = 25;
  double _humidity = 80;
  double _rainfall = 200;
  double _ph = 6.5;
  double _yieldLastSeason = 2500;
  double _fertilizerUsed = 120;
  String _season = 'Kharif';
  String _irrigation = 'Canal';
  String _previousCrop = 'Wheat';
  String _region = 'South';

  // Results
  PredictionResult? _result;
  bool _isPredicting = false;
  String? _predictionError;

  @override
  void dispose() {
    _pageController.dispose();
    super.dispose();
  }

  void _goToStep(int step) {
    setState(() => _currentStep = step);
    _pageController.animateToPage(
      step,
      duration: const Duration(milliseconds: 400),
      curve: Curves.easeInOut,
    );
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final picker = ImagePicker();
      // maxWidth/maxHeight required for BlueStacks gallery intent compatibility
      final XFile? picked = await picker.pickImage(
        source: source,
        maxWidth: 1024,
        maxHeight: 1024,
        imageQuality: 90,
      );
      if (picked != null) {
        final bytes = await picked.readAsBytes();
        setState(() {
          _imageBytes = bytes;
          _imageFileName = picked.name;
          _soilValidated = false;
          _validationError = null;
        });
        _validateSoilImage();
      }
    } catch (e) {
      _showSnackBar('Failed to pick image: ${e.toString()}', isError: true);
    }
  }

  Future<void> _validateSoilImage() async {
    if (_imageBytes == null) return;

    setState(() {
      _isValidatingSoil = true;
      _validationError = null;
    });

    try {
      final result =
          await ApiService.validateSoilImage(_imageBytes!, _imageFileName);
      setState(() {
        _isValidatingSoil = false;
        _soilValidated = result.isValid;
        _validationError = result.isValid ? null : result.message;
      });
    } on ApiException catch (e) {
      setState(() {
        _isValidatingSoil = false;
        _validationError = e.message;
      });
    }
  }

  Future<void> _runPrediction() async {
    if (_imageBytes == null || !_soilValidated) return;

    setState(() {
      _isPredicting = true;
      _predictionError = null;
    });

    try {
      final request = PredictionRequest(
        imageBytes: _imageBytes!,
        imageFileName: _imageFileName,
        nitrogen: _nitrogen,
        phosphorus: _phosphorus,
        potassium: _potassium,
        temperature: _temperature,
        humidity: _humidity,
        rainfall: _rainfall,
        ph: _ph,
        yieldLastSeason: _yieldLastSeason,
        fertilizerUsed: _fertilizerUsed,
        season: _season,
        irrigation: _irrigation,
        previousCrop: _previousCrop,
        region: _region,
      );

      final result = await ApiService.predict(request);
      setState(() {
        _result = result;
        _isPredicting = false;
      });
      _goToStep(2);
    } on ApiException catch (e) {
      setState(() {
        _isPredicting = false;
        _predictionError = e.message;
      });
      _showSnackBar(_predictionError!, isError: true);
    }
  }

  void _showSnackBar(String msg, {bool isError = false}) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Row(
          children: [
            Icon(
              isError ? Icons.error_outline : Icons.check_circle_outline,
              color: Colors.white,
              size: 20,
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                msg,
                style: GoogleFonts.workSans(fontSize: 13),
              ),
            ),
          ],
        ),
        backgroundColor: isError ? AppColors.danger : AppColors.primary,
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        margin: const EdgeInsets.all(16),
        duration: Duration(seconds: isError ? 4 : 2),
      ),
    );
  }

  void _resetAll() {
    setState(() {
      _imageBytes = null;
      _imageFileName = '';
      _soilValidated = false;
      _validationError = null;
      _result = null;
      _predictionError = null;
    });
    _goToStep(0);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: CustomScrollView(
        slivers: [
          // App bar — compact, with leading back nav
          SliverAppBar(
            expandedHeight: 80,
            pinned: true,
            backgroundColor: AppColors.primary,
            centerTitle: true,
            // Back button in leading position for step-by-step nav
            leading: _currentStep > 0
                ? IconButton(
                    icon: const Icon(Icons.arrow_back_rounded, size: 22),
                    tooltip: _currentStep == 2
                        ? 'Back to NPK & Details'
                        : 'Back to Soil Image',
                    onPressed: () => _goToStep(_currentStep - 1),
                  )
                : null,
            title: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                ClipRRect(
                  borderRadius: BorderRadius.circular(6),
                  child: Image.asset(
                    'assets/logo.jpg',
                    width: 24,
                    height: 24,
                    fit: BoxFit.cover,
                  ),
                ),
                const SizedBox(width: 10),
                Flexible(
                  child: Text(
                    'Precision Farming Assistant',
                    style: GoogleFonts.manrope(
                      fontSize: 16,
                      fontWeight: FontWeight.w800,
                      color: Colors.white,
                      letterSpacing: -0.3,
                    ),
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
              ],
            ),
            flexibleSpace: FlexibleSpaceBar(
              background: Container(
                decoration: const BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                    colors: [AppColors.primaryDark, AppColors.primary],
                  ),
                ),
              ),
            ),
            actions: [
              if (_currentStep > 0)
                IconButton(
                  icon: const Icon(Icons.refresh_rounded, size: 22),
                  tooltip: 'Start Over',
                  onPressed: _resetAll,
                ),
            ],
          ),

          // Step indicator
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
              child: StepIndicator(
                currentStep: _currentStep,
                steps: const [
                  'Soil Image',
                  'NPK & Details',
                  'Results',
                ],
              ),
            ),
          ),

          // Main content
          SliverFillRemaining(
            hasScrollBody: true,
            child: PageView(
              controller: _pageController,
              physics: const NeverScrollableScrollPhysics(),
              onPageChanged: (i) => setState(() => _currentStep = i),
              children: [
                // Step 1: Soil Image Upload
                SoilUploadCard(
                  imageBytes: _imageBytes,
                  isValidating: _isValidatingSoil,
                  isValidated: _soilValidated,
                  validationError: _validationError,
                  onPickCamera: () => _pickImage(ImageSource.camera),
                  onPickGallery: () => _pickImage(ImageSource.gallery),
                  onRemoveImage: () {
                    setState(() {
                      _imageBytes = null;
                      _imageFileName = '';
                      _soilValidated = false;
                      _validationError = null;
                    });
                  },
                  onContinue: _soilValidated ? () => _goToStep(1) : null,
                ),

                // Step 2: NPK & Environmental Inputs
                InputFormCard(
                  nitrogen: _nitrogen,
                  phosphorus: _phosphorus,
                  potassium: _potassium,
                  temperature: _temperature,
                  humidity: _humidity,
                  rainfall: _rainfall,
                  ph: _ph,
                  yieldLastSeason: _yieldLastSeason,
                  fertilizerUsed: _fertilizerUsed,
                  season: _season,
                  irrigation: _irrigation,
                  previousCrop: _previousCrop,
                  region: _region,
                  isPredicting: _isPredicting,
                  onNitrogenChanged: (v) =>
                      setState(() => _nitrogen = v),
                  onPhosphorusChanged: (v) =>
                      setState(() => _phosphorus = v),
                  onPotassiumChanged: (v) =>
                      setState(() => _potassium = v),
                  onTemperatureChanged: (v) =>
                      setState(() => _temperature = v),
                  onHumidityChanged: (v) =>
                      setState(() => _humidity = v),
                  onRainfallChanged: (v) =>
                      setState(() => _rainfall = v),
                  onPhChanged: (v) => setState(() => _ph = v),
                  onYieldChanged: (v) =>
                      setState(() => _yieldLastSeason = v),
                  onFertilizerChanged: (v) =>
                      setState(() => _fertilizerUsed = v),
                  onSeasonChanged: (v) =>
                      setState(() => _season = v),
                  onIrrigationChanged: (v) =>
                      setState(() => _irrigation = v),
                  onPreviousCropChanged: (v) =>
                      setState(() => _previousCrop = v),
                  onRegionChanged: (v) =>
                      setState(() => _region = v),
                  onSubmit: _runPrediction,
                ),

                // Step 3: Results — with back button to Step 2
                ResultView(
                  result: _result,
                  onNewAnalysis: _resetAll,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
