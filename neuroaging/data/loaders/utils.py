from pathlib import Path
import subprocess


def locate_anat_file(bids_dir: Path, extension: str = "nii.gz"):
    """
    Locate the anatomical file in a BIDS directory

    Parameters
    ----------
    bids_dir : Path
        Path to the BIDS directory
    """
    ce_corrected = list(bids_dir.glob(f"*ce-corrected*T1w.{extension}"))
    if len(ce_corrected) == 1:
        return ce_corrected[0]
    else:
        ce_uncorrected = list(bids_dir.glob(f"*ce-uncorrected*T1w.{extension}"))
        if len(ce_uncorrected) == 1:
            return ce_uncorrected[0]
        else:
            return None


def edit_cat12_template(cat12_template: Path, anat_file: Path):
    """
    Edit the CAT12 template

    Parameters
    ----------
    cat12_template : Path
        Path to the CAT12 template
    anat_file : Path
        Path to the anatomical file
    output_dir : Path
        Path to the output directory
    """
    with open(cat12_template, "r") as f:
        cat12_template_content = f.read()
    cat12_template_content = cat12_template_content.replace("SESSION_T1W", str(anat_file))
    # cat12_template_content = cat12_template_content.replace("ATLAS_NIFTI", str(atlases))
    return cat12_template_content


def rsync_gunzip(source: Path, destination: Path):
    """
    Rsync a directory

    Parameters
    ----------
    source : Path
        Source directory
    destination : Path
        Destination directory
    """

    subprocess.run(["rsync", "-azPL", str(source), str(destination)])
    subprocess.run(["gunzip", str(destination / source.name)])
