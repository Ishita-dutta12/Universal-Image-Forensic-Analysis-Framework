"""
UIFAF Main Pipeline

Universal Image Forensic Analysis Framework

Author:
Ishita Dutta
"""

from __future__ import annotations

import statistics
import statistics
import traceback
#from multiprocessing import context
#import statistics
#import statistics
import time
from pathlib import Path
import traceback

from matplotlib.style import context

#from matplotlib.style import context


#from httpx import request


from preprocessing.pixel_decoder import PixelDecoder

from metadata.metadata_parser import MetadataParser

from backbone.feature_extractor import FeatureExtractor

from attention.feature_enhancer import FeatureEnhancer
from decision.decision_engine import DecisionEngine
from report_generator.report_interpreter import ReportInterpreter
from report_generator.report_generator import ReportGenerator
from representation.spatial_encoder import SpatialEncoder
from representation.frequency_encoder import FrequencyEncoder
from representation.context_encoder import ContextEncoder
from representation.representation_fusion import RepresentationFusion
from representation.representation_models import (
    RepresentationOutput,
)
from visualization.visualization_engine import VisualizationEngine

from tests.test_report_formatter import report
from ufre.forensic_reasoner import (
    ForensicReasoner,
)
from visualization.visualization_engine import VisualizationEngine

from .pipeline_models import (
    PipelineContext,
    PipelineRequest,
    PipelineResult,
    PipelineStatistics,
)

from .pipeline_validator import (
    PipelineValidator,
)

from .pipeline_config import (
    PipelineConfig,
)
#from report_generator.report_models import ForensicReport
class UIFAFPipeline:
    """
    Main orchestration pipeline.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
    ) -> None:

        self.config = config or PipelineConfig()

        self.config.validate()

        #PipelineValidator.validate_config(
           # self.config
        #)

        self._initialize_modules()
    def _initialize_modules(self) -> None:

        # -------------------------
        # Preprocessing
        # -------------------------

        self.decoder = PixelDecoder()

        self.metadata_parser = MetadataParser()

        # -------------------------
        # Backbone
        # -------------------------

        self.backbone = FeatureExtractor(
            pretrained=True,
            device=self.config.device,
        )

        # -------------------------
        # Attention
        # -------------------------

        self.attention = FeatureEnhancer().to(self.config.device)

        # -------------------------
        # Representation
        # -------------------------

        stage_channels = {
            "stage1": 96,
            "stage2": 192,
            "stage3": 384,
            "stage4": 768,
        }

        self.spatial = SpatialEncoder(
            stage_channels
        ).to(self.config.device)

        self.frequency = FrequencyEncoder(
            stage_channels
        ).to(self.config.device)

        self.context = ContextEncoder(
            stage_channels
        ).to(self.config.device)

        self.representation_fusion = (
            RepresentationFusion(
                stage_channels
            ).to(self.config.device)
        )
        self.decision_engine = DecisionEngine().to(self.config.device)

        # -------------------------
        # Reasoning
        # -------------------------

        self.reasoner = (
    ForensicReasoner(
        stage_channels=stage_channels,
    ).to(self.config.device)
    
)
        self.report_interpreter = ReportInterpreter()

        self.report_generator = ReportGenerator()
        
        self.visualizer = VisualizationEngine(
    model=self.backbone.backbone.model,
    #target_layer=self.backbone.backbone.model.stages_3.blocks[2],
    output_directory="outputs",
)
    def _record_time(
        self,
        statistics: PipelineStatistics,
        module: str,
        start: float,
    ) -> None:

        statistics.execution_times[module] = (
            time.perf_counter() - start
        )

        statistics.completed_modules += 1
    def _new_statistics(
        self,
    ) -> PipelineStatistics:

        return PipelineStatistics()
    def _new_context(
        self,
    ) -> PipelineContext:

        return PipelineContext()
    
        # =====================================================
    # Decode Image
    # =====================================================

    def _decode_image(
        self,
        request: PipelineRequest,
        context: PipelineContext,
        statistics: PipelineStatistics,
    ) -> None:

        start = time.perf_counter()

        decoded = self.decoder.decode(
            request.image_path
        )

        context.decoded_image = decoded

        context.image_info = decoded.info

        self._record_time(
            statistics,
            "decoder",
            start,
        )

    # =====================================================
    # Parse Metadata
    # =====================================================

    def _parse_metadata(
        self,
        context: PipelineContext,
        statistics: PipelineStatistics,
    ) -> None:

        start = time.perf_counter()

        metadata = self.metadata_parser.parse(

            context.decoded_image.metadata

        )

        context.metadata = metadata

        self._record_time(

            statistics,

            "metadata",

            start,
        )

    # =====================================================
    # Backbone Feature Extraction
    # =====================================================

    def _extract_backbone_features(
    self,
    context: PipelineContext,
    statistics: PipelineStatistics,
) -> None:

        start = time.perf_counter()

        backbone = self.backbone(
        context.decoded_image
    )

        context.backbone = backbone

    # Store normalized tensor for Grad-CAM
        context.backbone_input = self.backbone.last_input_tensor

        self._record_time(
        statistics,
        "backbone",
        start,
    )

    # =====================================================
    # Attention Enhancement
    # =====================================================

    def _apply_attention(

        self,

        context: PipelineContext,

        statistics: PipelineStatistics,

    ) -> None:

        start = time.perf_counter()

        attention = self.attention(

            context.backbone

        )

        context.attention = attention

        self._record_time(

            statistics,

            "attention",

            start,
        )
        
        # =====================================================
    # Representation Learning
    # =====================================================

    def _build_representation(
        self,
        context: PipelineContext,
        statistics: PipelineStatistics,
    ) -> None:

        start = time.perf_counter()

        # -----------------------------------
        # Spatial Representation
        # -----------------------------------

        spatial_output = self.spatial(
            context.attention
        )

        # -----------------------------------
        # Frequency Representation
        # -----------------------------------

        frequency_output = self.frequency(
            context.attention
        )

        # -----------------------------------
        # Context Representation
        # -----------------------------------

        context_output = self.context(
            context.attention
        )

        # -----------------------------------
        # Adapter
        #
        # SpatialEncoder returns
        # RepresentationOutput
        #
        # RepresentationFusion expects
        # Dict[str, RepresentationFeature]
        #
        # No module changes required.
        # -----------------------------------

        spatial = spatial_output

        # -----------------------------------
        # Representation Fusion
        # -----------------------------------

        frequency = frequency_output
        context_features = context_output

        fused = self.representation_fusion(
    spatial,
    frequency,
    context_features,
)

        # -----------------------------------
        # Build RepresentationOutput
        # -----------------------------------

        representation = RepresentationOutput(

            stage1=fused["stage1"],

            stage2=fused["stage2"],

            stage3=fused["stage3"],

            stage4=fused["stage4"],

            original_size=context.attention.original_size,
        )

        representation.validate()

        context.representation = representation

        # ------------------------------------------------

        self._record_time(

            statistics,

            "representation",

            start,
        )
    
        # =====================================================
    # Universal Forensic Reasoner
    # =====================================================

    def _reason(
        self,
        context: PipelineContext,
        statistics: PipelineStatistics,
    ) -> None:

        start = time.perf_counter()

        reasoning = self.reasoner(
            context.representation
        )

        context.reasoning = reasoning

        self._record_time(
            statistics,
            "ufre",
            start,
        )
        
            # =====================================================
    # Run Pipeline
    # =====================================================

    def run(
        self,
        request: PipelineRequest,
    ) -> PipelineResult:

        statistics = self._new_statistics()

        context = self._new_context()
        pipeline_start = time.perf_counter()

        current_module = "decoder"

        try:

            # -----------------------------
            # Decode
            # -----------------------------
            PipelineValidator.validate_request(
                request
            )
            self._decode_image(
                request,
                context,
                statistics,
            )

            # -----------------------------
            # Metadata
            # -----------------------------
            current_module = "metadata"
            self._parse_metadata(
                context,
                statistics,
            )

            # -----------------------------
            # Backbone
            # -----------------------------
            current_module = "backbone"
            self._extract_backbone_features(
                context,
                statistics,
            )

            # -----------------------------
            # Attention
            # -----------------------------
            current_module = "attention"

            self._apply_attention(
                context,
                statistics,
            )

            # -----------------------------
            # Representation
            # -----------------------------
            current_module = "representation"
            self._build_representation(
                context,
                statistics,
            )

            # -----------------------------
            # UFRE
            # -----------------------------
            current_module = "ufre"
            self._reason(
                context,
                statistics,
            )
            
            # -----------------------------
# Visualization / Localization
# -----------------------------
            # -----------------------------
# Visualization / Localization
# -----------------------------
            # ----------------------------- 
            current_module = "visualization"

            start = time.perf_counter()

            context.localization = self.visualizer.generate(
            image_tensor=context.backbone_input,
            original_image=context.decoded_image.rgb_image,
            )

            self._record_time(
                statistics,
                "visualization",
                    start,
                )
            decision = self.decision_engine(
    reasoning=context.reasoning,
    metadata=context.metadata,
    localization=context.localization,
)

            report = self.report_interpreter.interpret(
                context=context,
                    decision=decision,
                    statistics=statistics,
                )

            # -------------------------------------
            # Store additional runtime information
            # -------------------------------------

            statistics.additional_info[
                "device"
            ] = str(self.config.device)

            if context.image_info is not None:
                statistics.additional_info["image_size"] = (
                    context.image_info.width,
                    context.image_info.height,
                )

                statistics.additional_info["filename"] = (
                    context.image_info.filename
                )
                statistics.additional_info["has_exif"] = (
                    context.image_info.has_exif
            )
                
                statistics.additional_info["metadata_fields"] = len(
                context.metadata.raw_exif
                ) 
                statistics.additional_info["camera_make"] = (
                    context.metadata.camera.make
                )

            statistics.additional_info["camera_model"] = (
                context.metadata.camera.model
            )

            statistics.additional_info["software"] = (
                context.metadata.camera.software
            )
            statistics.additional_info["total_time"] = (
                time.perf_counter() - pipeline_start
            )
            # -------------------------------------
            # NOTE
            #
            # Report generation is intentionally
            # skipped here.
            #
            # ReportGenerator requires many report
            # dataclasses which are not yet produced
            # by the current pipeline.
            #
            # The API layer (or CLI) should call
            # ReportGenerator after UFRE.
            # -------------------------------------

            return PipelineResult(
                context=context,
                statistics=statistics,
                report=report,
                success=True,
                message="Pipeline executed successfully.",
            )



        except Exception as exc:
            traceback.print_exc()   # <-- add this

            statistics.failed_module = current_module

            return PipelineResult(
        context=context,
        statistics=statistics,
        report=None,
        success=False,
        message=str(exc),
    )
    def analyze(
        self,
        image_path: str | Path,
    ) -> PipelineResult:

        request = PipelineRequest(
        image_path=Path(image_path)
    )

        return self.run(request)
    