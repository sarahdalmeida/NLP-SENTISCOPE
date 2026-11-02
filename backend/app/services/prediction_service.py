from backend.app.services.training_service import TrainingService, training_service


class PredictionService:
    def __init__(self, trainer: TrainingService) -> None:
        self.trainer = trainer

    def analyze(self, text: str, model_id: str) -> dict[str, object]:
        return self.trainer.predict(model_id, text)


prediction_service = PredictionService(training_service)
