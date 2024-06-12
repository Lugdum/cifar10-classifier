import os
import subprocess
import gdown

def download_and_extract_google_drive_file(file_id, output_dir):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    zip_path = os.path.join(output_dir, 'data.zip')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Télécharger le fichier avec gdown
    gdown.download(url, zip_path, quiet=False)
    
    # Extraire le fichier ZIP
    subprocess.run(['unzip', zip_path, '-d', output_dir], check=True)
    
    # Supprimer le fichier ZIP après extraction
    os.remove(zip_path)

def main():
    data_file_id = "1Mpi3rA9x4EoRntrpiV8H8LNv-jLsSwIn"
    models_file_id = "1IA2zQB1ik0DB8YBrqz4YW_vDXKFFTxYE"

    data_dir = "data"
    models_dir = "models"

    download_and_extract_google_drive_file(data_file_id, data_dir)
    download_and_extract_google_drive_file(models_file_id, models_dir)
    print("Download and extraction complete.")

if __name__ == "__main__":
    main()
