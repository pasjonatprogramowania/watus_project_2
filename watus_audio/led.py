class LEDStatusController:
    """
    Kontroler diod LED (lub innego wskaźnika wizualnego) do sygnalizowania stanu systemu.
    Obecnie implementacja jest pusta (dummy), ale przygotowana pod przyszłą rozbudowę.
    """
    
    def cleanup(self):
        """
        Sprząta zasoby (np. wyłącza diody) przy zamykaniu aplikacji.
        """
        pass

    def indicate_listening_state(self):
        """
        Sygnalizuje stan nasłuchiwania (np. kolor niebieski lub zielony).
        """
        pass

    def indicate_processing_state(self):
        """
        Sygnalizuje stan przetwarzania lub mówienia (np. pulsujący kolor lub czerwony).
        """
        pass
