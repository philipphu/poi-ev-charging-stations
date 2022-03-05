"""
Defines Paths class used across the package to handle filepaths.
"""
from loguru import logger
import pathlib


class Paths:
    """Class to reliably handle file paths relative to the root directory path.

    Attributes:
        root_path: The root path of the directory, which serves as the basis for all dynamically created filepaths.
    """

    def __init__(self) -> None:
        """Initializes Paths instance.

        Finds the root path as the directory two levels upwards of where this file is located.
        Prints out the detected root path.
        """
        self.root_path = pathlib.Path(__file__).resolve().parent.parent
        logger.info(f"Root path is detected to be {self.root_path}")

    @staticmethod
    def safe_return(path: pathlib.Path, mkdir: bool) -> pathlib.Path:
        """Safely return a path by optionally creating the parent directories to avoid errors when writing to the path.

        Args:
            path: Path to optionally create and return.
            mkdir: If True, creates the parent directories. If False, it has no effect.

        Returns:
            Input path.
        """
        if mkdir:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def general_output_file(self, filename: str, mkdir: bool) -> pathlib.Path:
        """Returns path to a general output file.

        Args:
            filename: Name of the output file.
            mkdir: If True, creates parent directories for safe writing. If False, won't create parents.

        Returns:
            Path to general output file.
        """
        return self.safe_return(
            self.root_path / "data" / "general_output" / filename,
            mkdir=mkdir,
        )

    def project_input_file(
        self, project: str, filename: str, mkdir: bool
    ) -> pathlib.Path:
        """Returns path to an input file for a certain project.

        Args:
            project: Name of the project.
            filename: Name of the input file.
            mkdir: If True, creates parent directories for safe writing. If False, won't create parents.

        Returns:
            Path to input file for a certain project.
        """
        return self.safe_return(
            self.root_path / "data" / project / "input" / filename,
            mkdir=mkdir,
        )

    def project_shape_file(self, project: str, mkdir: bool) -> pathlib.Path:
        """Returns path to the shape file for a certain project.

        Args:
            project: Name of the project.
            mkdir: If True, creates parent directories for safe writing. If False, won't create parents.

        Returns:
            Path to shape file for a certain project.
        """
        return self.safe_return(
            self.root_path
            / "data"
            / project
            / "input"
            / "shapefiles"
            / "shapefile.shp",
            mkdir=mkdir,
        )

    def project_output_file(
        self, project: str, filename: str, mkdir: bool
    ) -> pathlib.Path:
        """Returns path to an output file for a certain project.

        Args:
            project: Name of the project.
            filename: Name of the output file.
            mkdir: If True, creates parent directories for safe writing. If False, won't create parents.

        Returns:
            Path to output file for a certain project.
        """
        return self.safe_return(
            self.root_path / "data" / project / "output" / filename,
            mkdir=mkdir,
        )

    def project_output_dir(
        self,
        project: str,
    ) -> pathlib.Path:
        """Returns path to the output directory for a certain project.

        Args:
            project: Name of the project.

        Returns:
            Path to output directory for a certain project.
        """
        return self.safe_return(
            self.root_path / "data" / project / "output",
            mkdir=False,
        )

    def project_output_models_dir(
        self,
        project: str,
    ) -> pathlib.Path:
        """Returns path to the models output directory for a certain project.

        Args:
            project: Name of the project.

        Returns:
            Path to models output directory for a certain project.
        """
        return self.safe_return(
            self.root_path / "data" / project / "output" / "models",
            mkdir=False,
        )

    def project_output_model_file(
        self, project: str, model: str, filename: str, mkdir: bool
    ) -> pathlib.Path:
        """Returns path to an output file of a model for a certain project.

        Args:
            project: Name of the project.
            model: Name of the model.
            filename: Name of the output file.
            mkdir: If True, creates parent directories for safe writing. If False, won't create parents.

        Returns:
            Path to output file of a model for a certain project.
        """
        return self.safe_return(
            self.root_path / "data" / project / "output" / "models" / model / filename,
            mkdir=mkdir,
        )

    def project_output_model_dir(
        self,
        project: str,
        model: str,
    ) -> pathlib.Path:
        """Returns path to the output directory of a model for a certain project.

        Args:
            project: Name of the project.
            model: Name of the model.

        Returns:
            Path to output directory of a model for a certain project.
        """
        return self.safe_return(
            self.root_path / "data" / project / "output" / "models" / model,
            mkdir=False,
        )

    def www_directory_dir(
        self,
    ) -> pathlib.Path:
        """Returns path to the www writing directory."""
        return self.safe_return(
            self.root_path / "writing" / "poi_chargingpoints_www",
            mkdir=False,
        )