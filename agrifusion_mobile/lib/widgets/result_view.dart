import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:animate_do/animate_do.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:percent_indicator/circular_percent_indicator.dart';
import '../theme/app_theme.dart';
import '../models/prediction_models.dart';

class ResultView extends StatelessWidget {
  final PredictionResult? result;
  final VoidCallback onNewAnalysis;

  const ResultView({
    super.key,
    this.result,
    required this.onNewAnalysis,
  });

  @override
  Widget build(BuildContext context) {
    if (result == null) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.analytics_outlined, size: 64, color: AppColors.textMuted.withValues(alpha: 0.4)),
            const SizedBox(height: 16),
            Text('No results yet', style: GoogleFonts.manrope(fontSize: 18, fontWeight: FontWeight.w600, color: AppColors.textMuted)),
            const SizedBox(height: 8),
            Text('Complete Steps 1 & 2 to see predictions', style: GoogleFonts.workSans(fontSize: 14, color: AppColors.textMuted)),
          ],
        ),
      );
    }

    final r = result!;
    final soilColor = AppColors.soilColors[r.soilType] ?? AppColors.primary;

    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 32),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // ═══════════════════════════════════════════════════
          // SECTION 1: Soil Detection Result Banner
          // ═══════════════════════════════════════════════════
          FadeInDown(
            duration: const Duration(milliseconds: 500),
            child: Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                  colors: [AppColors.primaryDark, AppColors.primary],
                ),
                borderRadius: BorderRadius.circular(20),
                boxShadow: [
                  BoxShadow(color: AppColors.primary.withValues(alpha: 0.3), blurRadius: 20, offset: const Offset(0, 8)),
                ],
              ),
              child: Column(
                children: [
                  Text(
                    'RESULT ANALYSIS',
                    style: GoogleFonts.workSans(fontSize: 11, fontWeight: FontWeight.w600, color: Colors.white.withValues(alpha: 0.6), letterSpacing: 2),
                  ),
                  const SizedBox(height: 16),
                  Row(
                    children: [
                      CircularPercentIndicator(
                        radius: 44,
                        lineWidth: 7,
                        percent: (r.confidence / 100).clamp(0.0, 1.0),
                        center: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Text('${r.confidence.toStringAsFixed(1)}%', style: GoogleFonts.manrope(fontSize: 16, fontWeight: FontWeight.w800, color: Colors.white)),
                            Text('MATCH', style: GoogleFonts.workSans(fontSize: 8, fontWeight: FontWeight.w500, color: Colors.white.withValues(alpha: 0.6), letterSpacing: 1)),
                          ],
                        ),
                        progressColor: AppColors.primarySurface,
                        backgroundColor: Colors.white.withValues(alpha: 0.15),
                        circularStrokeCap: CircularStrokeCap.round,
                        animation: true,
                        animationDuration: 1200,
                      ),
                      const SizedBox(width: 16),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Container(
                              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                              decoration: BoxDecoration(color: soilColor.withValues(alpha: 0.3), borderRadius: BorderRadius.circular(6)),
                              child: Text('DETECTED SOIL', style: GoogleFonts.workSans(fontSize: 9, fontWeight: FontWeight.w600, color: Colors.white, letterSpacing: 1.5)),
                            ),
                            const SizedBox(height: 6),
                            Text(r.soilType, style: GoogleFonts.manrope(fontSize: 22, fontWeight: FontWeight.w800, color: Colors.white)),
                            const SizedBox(height: 4),
                            Text('Multimodal Fusion Analysis', style: GoogleFonts.workSans(fontSize: 10, color: Colors.white.withValues(alpha: 0.5))),
                          ],
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),

          const SizedBox(height: 16),

          // ═══════════════════════════════════════════════════
          // SECTION 2: Soil Fertilizer Recommendation
          // ═══════════════════════════════════════════════════
          FadeInUp(
            delay: const Duration(milliseconds: 200),
            duration: const Duration(milliseconds: 500),
            child: Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: AppColors.surface,
                borderRadius: BorderRadius.circular(16),
                boxShadow: [BoxShadow(color: Colors.black.withValues(alpha: 0.04), blurRadius: 12, offset: const Offset(0, 4))],
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.all(8),
                        decoration: BoxDecoration(color: AppColors.primarySurfaceLight, borderRadius: BorderRadius.circular(10)),
                        child: const Icon(Icons.science_rounded, color: AppColors.primary, size: 20),
                      ),
                      const SizedBox(width: 12),
                      Text('Soil Fertilizer Advice', style: GoogleFonts.manrope(fontSize: 16, fontWeight: FontWeight.w700, color: AppColors.textPrimary)),
                    ],
                  ),
                  const SizedBox(height: 14),
                  _buildInfoRow('Recommended', r.soilFertilizer, Icons.eco_rounded),
                  const SizedBox(height: 8),
                  _buildInfoRow('NPK Ratio', 'N:P:K = ${r.soilNpk}', Icons.balance_rounded),
                ],
              ),
            ),
          ),

          const SizedBox(height: 16),

          // ═══════════════════════════════════════════════════
          // SECTION 3: Crop Recommendations
          // ═══════════════════════════════════════════════════
          FadeInUp(
            delay: const Duration(milliseconds: 300),
            duration: const Duration(milliseconds: 500),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 4),
                  child: Text('RECOMMENDED CROPS', style: GoogleFonts.workSans(fontSize: 12, fontWeight: FontWeight.w600, color: AppColors.textMuted, letterSpacing: 1.5)),
                ),
                const SizedBox(height: 12),
                ...r.recommendedCrops.map((crop) => _buildCropCard(crop, soilColor, r.soilType)),
              ],
            ),
          ),

          const SizedBox(height: 16),

          // ═══════════════════════════════════════════════════
          // SECTION 4: NPK Forecast Banner
          // ═══════════════════════════════════════════════════
          FadeInUp(
            delay: const Duration(milliseconds: 400),
            duration: const Duration(milliseconds: 500),
            child: Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                gradient: const LinearGradient(begin: Alignment.topLeft, end: Alignment.bottomRight, colors: [Color(0xFF1A3A2A), Color(0xFF0D3320)]),
                borderRadius: BorderRadius.circular(14),
              ),
              child: Row(
                children: [
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('NPK FORECAST', style: GoogleFonts.workSans(fontSize: 9, fontWeight: FontWeight.w600, color: Colors.white.withValues(alpha: 0.5), letterSpacing: 1.5)),
                        const SizedBox(height: 6),
                        RichText(
                          text: TextSpan(
                            style: GoogleFonts.manrope(fontSize: 14, fontWeight: FontWeight.w600, color: Colors.white),
                            children: [
                              const TextSpan(text: 'Soil balance trending towards '),
                              TextSpan(text: 'Hyper-Fertile', style: GoogleFonts.manrope(fontWeight: FontWeight.w800, color: AppColors.primarySurface)),
                              const TextSpan(text: ' next season.'),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(width: 12),
                  Container(
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(color: Colors.white.withValues(alpha: 0.1), borderRadius: BorderRadius.circular(12)),
                    child: const Icon(Icons.trending_up_rounded, color: AppColors.primarySurface, size: 28),
                  ),
                ],
              ),
            ),
          ),

          const SizedBox(height: 16),

          // ═══════════════════════════════════════════════════
          // SECTION 5: Probability Chart
          // ═══════════════════════════════════════════════════
          if (r.allProbabilities.isNotEmpty)
            FadeInUp(
              delay: const Duration(milliseconds: 500),
              duration: const Duration(milliseconds: 500),
              child: Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: AppColors.surface,
                  borderRadius: BorderRadius.circular(16),
                  boxShadow: [BoxShadow(color: Colors.black.withValues(alpha: 0.04), blurRadius: 12, offset: const Offset(0, 4))],
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Container(padding: const EdgeInsets.all(8), decoration: BoxDecoration(color: AppColors.primarySurfaceLight, borderRadius: BorderRadius.circular(10)), child: const Icon(Icons.bar_chart_rounded, color: AppColors.primary, size: 20)),
                        const SizedBox(width: 12),
                        Text('Soil Type Probabilities', style: GoogleFonts.manrope(fontSize: 16, fontWeight: FontWeight.w700, color: AppColors.textPrimary)),
                      ],
                    ),
                    const SizedBox(height: 16),
                    SizedBox(height: 200, child: _buildBarChart(r)),
                  ],
                ),
              ),
            ),

          const SizedBox(height: 24),

          // ═══════════════════════════════════════════════════
          // ACTION BUTTON: New Analysis
          // ═══════════════════════════════════════════════════
          FadeInUp(
            delay: const Duration(milliseconds: 600),
            duration: const Duration(milliseconds: 400),
            child: ElevatedButton.icon(
              onPressed: onNewAnalysis,
              icon: const Icon(Icons.refresh_rounded, size: 20),
              label: Text('New Analysis', style: GoogleFonts.manrope(fontSize: 15, fontWeight: FontWeight.w700)),
              style: ElevatedButton.styleFrom(
                backgroundColor: AppColors.primary,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
                elevation: 0,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildInfoRow(String label, String value, IconData icon) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(color: AppColors.surfaceVariant, borderRadius: BorderRadius.circular(12)),
      child: Row(
        children: [
          Icon(icon, size: 18, color: AppColors.primary),
          const SizedBox(width: 12),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(label.toUpperCase(), style: GoogleFonts.workSans(fontSize: 10, fontWeight: FontWeight.w600, color: AppColors.textMuted, letterSpacing: 0.8)),
              const SizedBox(height: 2),
              Text(value, style: GoogleFonts.manrope(fontSize: 14, fontWeight: FontWeight.w700, color: AppColors.textPrimary)),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildCropCard(CropRecommendation crop, Color soilColor, String soilType) {
    final isTop = crop.rank == 1;
    final soilBrightness = ThemeData.estimateBrightnessForColor(soilColor);
    final topTextColor = soilBrightness == Brightness.dark ? Colors.white : const Color(0xFF1B1C1A);
    final topMutedColor = soilBrightness == Brightness.dark ? Colors.white.withValues(alpha: 0.7) : const Color(0xFF404942);

    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: isTop ? soilColor.withValues(alpha: 0.3) : Colors.black.withValues(alpha: 0.04),
            blurRadius: isTop ? 16 : 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Soil-colored banner for Rank #1
            if (isTop)
              Container(
                padding: const EdgeInsets.fromLTRB(18, 16, 18, 14),
                color: soilColor,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                          decoration: BoxDecoration(color: Colors.white.withValues(alpha: 0.2), borderRadius: BorderRadius.circular(8)),
                          child: Text('RANK #1 · HIGHLY RECOMMENDED', style: GoogleFonts.workSans(fontSize: 9, fontWeight: FontWeight.w700, color: topTextColor, letterSpacing: 0.5)),
                        ),
                        const Spacer(),
                        Row(children: List.generate(5, (i) => Icon(i < crop.stars ? Icons.star_rounded : Icons.star_outline_rounded, size: 16, color: i < crop.stars ? const Color(0xFFFFD700) : topMutedColor.withValues(alpha: 0.4)))),
                      ],
                    ),
                    const SizedBox(height: 10),
                    Text(crop.name, style: GoogleFonts.manrope(fontSize: 28, fontWeight: FontWeight.w800, color: topTextColor)),
                    const SizedBox(height: 4),
                    Text('Based on $soilType properties • ${crop.stars * 20}% match', style: GoogleFonts.workSans(fontSize: 11, color: topMutedColor)),
                  ],
                ),
              ),
            // Fertilizer details
            Container(
              padding: const EdgeInsets.all(16),
              color: AppColors.surface,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  if (!isTop) ...[
                    Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                          decoration: BoxDecoration(color: AppColors.surfaceDim, borderRadius: BorderRadius.circular(8)),
                          child: Text('RANK #${crop.rank}', style: GoogleFonts.workSans(fontSize: 10, fontWeight: FontWeight.w700, color: AppColors.textSecondary, letterSpacing: 0.5)),
                        ),
                        const Spacer(),
                        Row(children: List.generate(5, (i) => Icon(i < crop.stars ? Icons.star_rounded : Icons.star_outline_rounded, size: 14, color: i < crop.stars ? const Color(0xFFFFB300) : AppColors.surfaceDim))),
                      ],
                    ),
                    const SizedBox(height: 10),
                    Text(crop.name, style: GoogleFonts.manrope(fontSize: 22, fontWeight: FontWeight.w800, color: AppColors.textPrimary)),
                    const SizedBox(height: 12),
                  ],
                  if (isTop)
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                      margin: const EdgeInsets.only(bottom: 10),
                      decoration: BoxDecoration(color: AppColors.primarySurfaceLight, borderRadius: BorderRadius.circular(8)),
                      child: Row(
                        children: [
                          const Icon(Icons.auto_awesome_rounded, size: 14, color: AppColors.primary),
                          const SizedBox(width: 6),
                          Text('SCIENTIFIC FERTILIZER RECOMMENDATION', style: GoogleFonts.workSans(fontSize: 9, fontWeight: FontWeight.w700, color: AppColors.primary, letterSpacing: 0.8)),
                        ],
                      ),
                    ),
                  Container(
                    padding: const EdgeInsets.all(14),
                    decoration: BoxDecoration(color: AppColors.surfaceVariant, borderRadius: BorderRadius.circular(12)),
                    child: Column(
                      children: [
                        Row(
                          children: [
                            const Icon(Icons.science_outlined, size: 16, color: AppColors.primaryLight),
                            const SizedBox(width: 8),
                            Text('RECOMMENDED TYPE', style: GoogleFonts.workSans(fontSize: 10, fontWeight: FontWeight.w600, color: AppColors.textMuted, letterSpacing: 0.5)),
                            const Spacer(),
                            Flexible(child: Text(crop.fertilizer, style: GoogleFonts.manrope(fontSize: 13, fontWeight: FontWeight.w700, color: AppColors.primary), overflow: TextOverflow.ellipsis)),
                          ],
                        ),
                        const Divider(height: 20),
                        Row(
                          children: [
                            const Icon(Icons.balance_rounded, size: 16, color: AppColors.primaryLight),
                            const SizedBox(width: 8),
                            Text('RATIO (N:P:K)', style: GoogleFonts.workSans(fontSize: 10, fontWeight: FontWeight.w600, color: AppColors.textMuted, letterSpacing: 0.5)),
                            const Spacer(),
                            Text(crop.npk, style: GoogleFonts.manrope(fontSize: 13, fontWeight: FontWeight.w700, color: AppColors.textPrimary)),
                          ],
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildBarChart(PredictionResult r) {
    final entries = r.allProbabilities.entries.toList()..sort((a, b) => b.value.compareTo(a.value));
    return BarChart(
      BarChartData(
        alignment: BarChartAlignment.spaceAround,
        maxY: 100,
        barTouchData: BarTouchData(
          enabled: true,
          touchTooltipData: BarTouchTooltipData(
            getTooltipItem: (group, groupIndex, rod, rodIndex) {
              final name = entries[group.x.toInt()].key;
              return BarTooltipItem('$name\n${rod.toY.toStringAsFixed(1)}%', GoogleFonts.workSans(color: Colors.white, fontWeight: FontWeight.w600, fontSize: 12));
            },
          ),
        ),
        titlesData: FlTitlesData(
          show: true,
          bottomTitles: AxisTitles(sideTitles: SideTitles(showTitles: true, reservedSize: 40, getTitlesWidget: (value, meta) {
            final idx = value.toInt();
            if (idx >= 0 && idx < entries.length) {
              final short = entries[idx].key.replaceAll(' Soil', '').substring(0, entries[idx].key.replaceAll(' Soil', '').length.clamp(0, 6));
              return Padding(padding: const EdgeInsets.only(top: 8), child: Text(short, style: GoogleFonts.workSans(fontSize: 9, fontWeight: FontWeight.w500, color: AppColors.textMuted)));
            }
            return const SizedBox();
          })),
          leftTitles: AxisTitles(sideTitles: SideTitles(showTitles: true, reservedSize: 35, getTitlesWidget: (value, meta) {
            if (value % 25 == 0) return Text('${value.toInt()}%', style: GoogleFonts.workSans(fontSize: 9, color: AppColors.textMuted));
            return const SizedBox();
          })),
          topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
          rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
        ),
        gridData: FlGridData(show: true, drawVerticalLine: false, horizontalInterval: 25, getDrawingHorizontalLine: (value) => FlLine(color: AppColors.surfaceDim.withValues(alpha: 0.5), strokeWidth: 1)),
        borderData: FlBorderData(show: false),
        barGroups: List.generate(entries.length, (i) {
          final barColor = AppColors.soilColors[entries[i].key] ?? AppColors.chartColors[i % AppColors.chartColors.length];
          return BarChartGroupData(x: i, barRods: [
            BarChartRodData(toY: entries[i].value, color: barColor, width: 22, borderRadius: const BorderRadius.vertical(top: Radius.circular(6)), backDrawRodData: BackgroundBarChartRodData(show: true, toY: 100, color: AppColors.surfaceVariant)),
          ]);
        }),
      ),
      duration: const Duration(milliseconds: 800),
    );
  }
}
