"""
Tests for report_models.py
"""

from datetime import datetime

from report_generator.report_models import (
    ConfidenceAssessment,
    ConfidenceLevel,
    EvidenceScores,
    EvidenceSummary,
    FileInformation,
    FinalVerdict,
    Finding,
    ForensicReport,
    LocalizationResult,
    MetadataSummary,
    PerformanceStatistics,
    RawModelOutput,
    RawModelOutputs,
    Recommendation,
    RecommendationCategory,
    RecommendationPriority,
    ReportLevel,
    RiskAssessment,
    RiskLevel,
    SystemInformation,
    ThresholdInformation,
)
def test_file_information():

    info = FileInformation(
        filename="image.jpg",
        file_size_bytes=2048,
        width=512,
        height=512,
        image_format="JPEG",
        channels=3,
        color_space="RGB",
        resolution="512x512",
        extension=".jpg",
    )

    assert info.filename == "image.jpg"
    assert info.width == 512
    assert info.height == 512
    assert info.channels == 3

def test_evidence_scores():

    scores = EvidenceScores(
        scores={
            "Authentic": 0.95,
            "AI": 0.05,
        }
    )

    assert scores.scores["Authentic"] == 0.95
def test_evidence_summary():

    summary = EvidenceSummary(
        spatial_score=0.9,
        frequency_score=0.8,
        context_score=0.7,
        metadata_score=0.6,
        overall_score=0.85,
    )

    assert summary.overall_score == 0.85
    
def test_risk_assessment():

    risk = RiskAssessment(
        risk_score=0.15,
        risk_level=RiskLevel.LOW,
    )

    assert risk.risk_score == 0.15
    assert risk.risk_level == RiskLevel.LOW
def test_confidence_assessment():

    confidence = ConfidenceAssessment(
        confidence=0.96,
        uncertainty=0.04,
        confidence_level=ConfidenceLevel.VERY_HIGH,
    )

    assert confidence.confidence == 0.96
    assert confidence.uncertainty == 0.04
    assert confidence.confidence_level == ConfidenceLevel.VERY_HIGH
def test_finding():

    finding = Finding(
        title="GAN Artifact",
        description="Periodic GAN artifacts detected.",
        confidence=0.97,
        source_module="Representation",
        source_branch="Frequency",
        supporting_evidence=["FFT", "Wavelet"],
    )

    assert finding.title == "GAN Artifact"
    assert len(finding.supporting_evidence) == 2
def test_recommendation():

    recommendation = Recommendation(
        message="Verify manually.",
        priority=RecommendationPriority.HIGH,
        category=RecommendationCategory.SECURITY,
    )

    assert recommendation.message == "Verify manually."
def test_localization():

    localization = LocalizationResult(
        manipulated_area_percentage=25.4,
        detected_regions=3,
        peak_confidence=0.94,
    )

    assert localization.detected_regions == 3
    
def test_metadata_summary():

    metadata = MetadataSummary(
        metadata={
            "Camera": "Canon",
            "ISO": "100",
        },
        exif_present=True,
        metadata_modified=False,
        suspicious_fields=["Software"],
        software="Photoshop",
        camera_model="Canon EOS R5",
        gps_present=False,
    )

    assert metadata.exif_present is True
    assert metadata.metadata["Camera"] == "Canon"
    assert metadata.software == "Photoshop"
def test_raw_model_output():

    output = RawModelOutput(
        module="Backbone",
        branch="Spatial",
        prediction="Authentic",
        probability=0.98,
        confidence=0.96,
        threshold=0.50,
    )

    assert output.module == "Backbone"
    assert output.branch == "Spatial"
    assert output.prediction == "Authentic"
def test_raw_model_outputs():

    outputs = RawModelOutputs(
        outputs=[
            RawModelOutput(
                module="Spatial",
                branch="RGB",
                prediction="Authentic",
                probability=0.91,
                confidence=0.90,
                threshold=0.50,
            ),
            RawModelOutput(
                module="Frequency",
                branch="Wavelet",
                prediction="Authentic",
                probability=0.94,
                confidence=0.93,
                threshold=0.50,
            ),
        ]
    )

    assert len(outputs.outputs) == 2
def test_threshold_information():

    thresholds = ThresholdInformation(
        thresholds={
            "manipulation": 0.50,
            "confidence": 0.80,
            "localization": 0.60,
        }
    )

    assert thresholds.thresholds["manipulation"] == 0.50
def test_system_information():

    system = SystemInformation(
        framework_name="UIFAF",
        framework_version="1.0.0",
        backbone="ConvNeXtV2",
        attention_module="EMA",
        reasoning_engine="UFRE",
        python_version="3.11",
        device="CPU",
        model_checkpoint="checkpoint.pt",
    )

    assert system.framework_name == "UIFAF"
    assert system.backbone == "ConvNeXtV2"
def test_performance_statistics():

    performance = PerformanceStatistics(
        preprocessing_time_ms=5.2,
        feature_extraction_time_ms=18.6,
        reasoning_time_ms=7.3,
        total_inference_time_ms=31.1,
    )

    assert performance.total_inference_time_ms == 31.1
def build_report():

    return ForensicReport(
        report_id="REPORT-001",
        report_level=ReportLevel.DETAILED,
        timestamp=datetime.now(),

        file_information=FileInformation(
            filename="image.jpg",
            file_size_bytes=1024,
            width=512,
            height=512,
            image_format="JPEG",
            channels=3,
            color_space="RGB",
            resolution="512x512",
            extension=".jpg",
        ),

        evidence_scores=EvidenceScores(
            scores={
                "Authentic": 0.92,
                "Manipulated": 0.08,
            }
        ),

        evidence_summary=EvidenceSummary(
            spatial_score=0.92,
            frequency_score=0.90,
            context_score=0.91,
            metadata_score=0.89,
            overall_score=0.91,
        ),

        risk_assessment=RiskAssessment(
            risk_score=0.08,
            risk_level=RiskLevel.LOW,
        ),

        confidence_assessment=ConfidenceAssessment(
            confidence=0.96,
            uncertainty=0.04,
            confidence_level=ConfidenceLevel.VERY_HIGH,
        ),

        final_verdict=FinalVerdict.AUTHENTIC,

        findings=[
            Finding(
                title="No forgery",
                description="No manipulation detected.",
                confidence=0.95,
                source_module="UFRE",
                source_branch="Fusion",
            )
        ],

        recommendations=[
            Recommendation(
                message="Image appears authentic.",
                priority=RecommendationPriority.LOW,
                category=RecommendationCategory.GENERAL,
            )
        ],

        localization=LocalizationResult(
            manipulated_area_percentage=0.0,
            detected_regions=0,
            peak_confidence=0.0,
        ),

        metadata_summary=MetadataSummary(
    metadata={
        "Camera": "Canon",
        "ISO": "100",
    },
    exif_present=True,
    metadata_modified=False,
    suspicious_fields=[],
    software="Canon Utility",
    camera_model="Canon EOS R5",
    gps_present=False,
),

        raw_outputs=RawModelOutputs(
            outputs=[]
        ),

        threshold_information=ThresholdInformation(
            thresholds={
                "manipulation": 0.50,
                "confidence": 0.80,
                "localization": 0.60,
            }
        ),

        system_information=SystemInformation(
    framework_name="UIFAF",
    framework_version="1.0.0",
    backbone="ConvNeXtV2",
    attention_module="EMA",
    reasoning_engine="UFRE",
    python_version="3.11.9",
    device="CPU",
    model_checkpoint="convnextv2.pth",
),

        performance_statistics=PerformanceStatistics(
            preprocessing_time_ms=5,
            feature_extraction_time_ms=15,
            reasoning_time_ms=10,
            total_inference_time_ms=30,
        ),
    )
    
def test_forensic_report_creation():

    report = build_report()

    assert report.report_id == "REPORT-001"
    assert report.final_verdict == FinalVerdict.AUTHENTIC
def test_is_authentic():

    report = build_report()

    assert report.is_authentic is True
    
def test_is_authentic():

    report = build_report()

    assert report.is_authentic is True
    
def test_is_manipulated():

    report = build_report()

    assert report.is_manipulated is False
def test_confidence_percentage():

    report = build_report()

    assert report.confidence_percentage == 96.0
def test_risk_percentage():

    report = build_report()

    assert report.risk_percentage == 8.0
def test_total_findings():

    report = build_report()

    assert report.total_findings == 1
def test_total_recommendations():

    report = build_report()

    assert report.total_recommendations == 1
    
def test_has_metadata():

    report = build_report()

    assert report.has_metadata is True
def test_has_localization():

    report = build_report()

    assert report.has_localization is True
    
def test_has_raw_outputs():

    report = build_report()

    assert report.has_raw_outputs is True
    
def test_inference_time():

    report = build_report()

    assert report.inference_time_ms == 30