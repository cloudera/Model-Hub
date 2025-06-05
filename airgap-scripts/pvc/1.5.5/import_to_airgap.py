#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
import logging

import yaml
from typing import List, Dict, Optional, Tuple

# courtesy : https://stackoverflow.com/questions/43765849/pyyaml-load-and-dump-yaml-file-and-preserve-tags-customtag
class SafeUnknownConstructor(yaml.constructor.SafeConstructor):
    def __init__(self):
        yaml.constructor.SafeConstructor.__init__(self)

    def construct_undefined(self, node):
        data = getattr(self, 'construct_' + node.id)(node)
        datatype = type(data)
        wraptype = type('TagWrap_'+datatype.__name__, (datatype,), {})
        wrapdata = wraptype(data)
        wrapdata.tag = lambda: None
        wrapdata.datatype = lambda: None
        setattr(wrapdata, "wrapTag", node.tag)
        setattr(wrapdata, "wrapType", datatype)
        return wrapdata


class SafeUnknownLoader(SafeUnknownConstructor, yaml.loader.SafeLoader):

    def __init__(self, stream):
        SafeUnknownConstructor.__init__(self)
        yaml.loader.SafeLoader.__init__(self, stream)


class SafeUnknownRepresenter(yaml.representer.SafeRepresenter):
    def represent_data(self, wrapdata):
        tag = False
        if type(wrapdata).__name__.startswith('TagWrap_'):
            datatype = getattr(wrapdata, "wrapType")
            tag = getattr(wrapdata, "wrapTag")
            data = datatype(wrapdata)
        else:
            data = wrapdata
        node = super(SafeUnknownRepresenter, self).represent_data(data)
        if tag:
            node.tag = tag
        return node

class SafeUnknownDumper(SafeUnknownRepresenter, yaml.dumper.SafeDumper):

    def __init__(self, stream,
            default_style=None, default_flow_style=False,
            canonical=None, indent=None, width=None,
            allow_unicode=None, line_break=None,
            encoding=None, explicit_start=None, explicit_end=None,
            version=None, tags=None, sort_keys=True):

        SafeUnknownRepresenter.__init__(self, default_style=default_style,
                default_flow_style=default_flow_style, sort_keys=sort_keys)

        yaml.dumper.SafeDumper.__init__(self,  stream,
                                        default_style=default_style,
                                        default_flow_style=default_flow_style,
                                        canonical=canonical,
                                        indent=indent,
                                        width=width,
                                        allow_unicode=allow_unicode,
                                        line_break=line_break,
                                        encoding=encoding,
                                        explicit_start=explicit_start,
                                        explicit_end=explicit_end,
                                        version=version,
                                        tags=tags,
                                        sort_keys=sort_keys)


MySafeLoader = SafeUnknownLoader
yaml.constructor.SafeConstructor.add_constructor(None, SafeUnknownConstructor.construct_undefined)


class ModelParser:
    def __init__(self, yaml_file_path: str):
        """Initialize the parser with a YAML file path."""
        self.yaml_file_path = yaml_file_path
        self.models_data = self._load_yaml()
    
    def _load_yaml(self) -> Dict:
        """Load and parse the YAML file."""
        try:
            with open(self.yaml_file_path, "r") as file:
                yaml_data = file.read()
                return yaml.load(yaml_data,Loader=MySafeLoader)
        except FileNotFoundError:
            print(f"Error: File {self.yaml_file_path} not found.")
            return {}
        except yaml.YAMLError as e:
            print(f"Error parsing YAML: {e}")
            return {}
    
    def get_all_models(self) -> List[Dict[str, str]]:
        """Fetch all models with their basic information."""
        models = []
        if 'models' in self.models_data:
            for model in self.models_data['models']:
                model_info = {
                    'name': model.get('name', ''),
                    'displayName': model.get('displayName', ''),
                    'modelHubID': model.get('modelHubID', ''),
                    'category': model.get('category', ''),
                    'description': model.get('description', '')
                }
                models.append(model_info)
        return models
    
    def get_model_variant_ids(self, model_name: str) -> List[str]:
        """
        Fetch all model variant IDs for a given model name.
        
        Args:
            model_name: The name of the model to search for
            
        Returns:
            List of variant IDs for the specified model
        """
        variant_ids = []
        if 'models' in self.models_data:
            for model in self.models_data['models']:
                if model.get('name', '').lower() == model_name.lower():
                    if 'modelVariants' in model:
                        for variant in model['modelVariants']:
                            variant_id = variant.get('variantId', '')
                            if variant_id:
                                variant_ids.append(variant_id)
                    break
        return variant_ids
    
    def get_optimization_profile_ids(self, model_name: str, variant_id: str = None) -> List[str]:
        """
        Fetch all optimization profile IDs for a given model name and optional variant ID.
        
        Args:
            model_name: The name of the model
            variant_id: Optional variant ID. If None, gets profiles from all variants
            
        Returns:
            List of optimization profile IDs
        """
        profile_ids = []
        if 'models' in self.models_data:
            for model in self.models_data['models']:
                if model.get('name', '').lower() == model_name.lower():
                    if 'modelVariants' in model:
                        for variant in model['modelVariants']:
                            # If variant_id is specified, only process that variant
                            if variant_id and variant.get('variantId', '').lower() != variant_id.lower():
                                continue
                            
                            if 'optimizationProfiles' in variant:
                                for profile in variant['optimizationProfiles']:
                                    profile_id = profile.get('profileId', '')
                                    if profile_id:
                                        profile_ids.append(profile_id)
                    break
        return profile_ids
    
    def get_detailed_model_info(self, model_name: str) -> Optional[Dict]:
        """
        Get detailed information about a specific model including all variants and profiles.
        
        Args:
            model_name: The name of the model
            
        Returns:
            Dictionary containing detailed model information
        """
        if 'models' in self.models_data:
            for model in self.models_data['models']:
                if model.get('name', '').lower() == model_name.lower():
                    return model
        return None
    




def extract_last_part(input_string):
    # Split the string by '/' and return the last part
    return input_string.split('/')[-1]

def run_ngc_info_command(ID):
    logging.info(f"Start Fetching model info from NGC: {ID}")
    cmd = ["ngc", "registry", "model", "info", ID, "--format_type", "json"]
    
    # print the command without exposing Cloudera's key
    logging.info(cmd)
    
    # Set up environment with API keys
    # env = os.environ.copy()
    # env["NGC_CLI_API_KEY"] = os.environ.get("NGC_API_KEY")
    # env["NGC_CLI_ORG"] = os.environ.get("NGC_CLI_ORG")
    
    # Run the command and collect output
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=False,
            check=True
        )
        
        logging.info(f"Finish Fetching NGC model repo {ID}")
        
        # Parse the JSON output
        try:
            metadata_map = json.loads(result.stdout)
            return metadata_map, None
        except json.JSONDecodeError as e:
            logging.error(f"Error while Unmarshalling the NGC info command to map: {str(e)}")
            logging.error(f"ngc model info stdout: {result.stdout.decode()}")
            logging.error(f"ngc model info stderr: {result.stderr.decode()}")
            return None, e
            
    except subprocess.CalledProcessError as e:
        logging.error(f"Error Fetching model repo: {ID}")
        logging.error(f"Error: {str(e)}")
        logging.error(f"Command stdout: {e.stdout.decode() if e.stdout else ''}")
        logging.error(f"Command stderr: {e.stderr.decode() if e.stderr else ''}")
        return None, Exception(e.stderr.decode() if e.stderr else str(e))



def get_ngc_model_info(model_id, tag):
    model_metadata_map, err = run_ngc_info_command(model_id)
    if err:
        logging.error(f"Error Fetching model repo: {model_id}")
        logging.error(f"Error: {str(err)}")
        return ""
    versionMetadataMap, err = run_ngc_info_command(model_id + ":" + tag)
    if err:
        logging.error(f"Error Fetching model repo: {model_id}")
        logging.error(f"Error: {str(err)}")
        return ""
    v1 = versionMetadataMap.get("totalSizeInBytes", "")
    v2 = model_metadata_map.get("versionId", "")
    if v2 != "":
        model_metadata_map["versionId"] = v2
    return model_metadata_map


def load_ngc_spec(spec_file):
    with open(spec_file, "r") as file:
        yaml_data = file.read()

    yaml_data = yaml.load(yaml_data,Loader=MySafeLoader)
    return yaml_data


def execute_ngc_download_command(repo_id, folder_location, files=None):
    """
    Execute NGC download command to download model files.
    
    Args:
        repo_id (str): Repository ID
        folder_location (str): Destination folder
        files (list, optional): List of files to download. If None, download the entire repo.
        
    Returns:
        tuple: (folder_location, error)
    """
    # Set environment variables
    # env = os.environ.copy()
    # env["NGC_CLI_API_KEY"] = os.environ.get("NGC_API_KEY")
    # env["NGC_CLI_ORG"] = os.environ.get("NGC_CLI_ORG")
    ngcPrefix = 'ngc://'
    if repo_id.startswith(ngcPrefix):
        repo_id = repo_id[len(ngcPrefix):]
    if files and len(files) > 0:
        # Download specific files
        for file in files:
            cmd = [
                "ngc", "registry", "model", "download-version", 
                repo_id, "--file", file, "--dest", folder_location, 
                "--format_type", "json"
            ]
            logging.info(cmd)
            
            try:
                # output = subprocess.check_output(cmd,env=env, stderr=subprocess.STDOUT)
                output = subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                logging.error(f"Error download model repo: {repo_id}")
                logging.error(f"Error: {str(e)}")
                return folder_location, e
    else:
        # Download entire repository
        cmd = [
            "ngc", "registry", "model", "download-version", 
            repo_id, "--dest", folder_location, "--format_type", "json"
        ]
        logging.info(cmd)
        
        try:
            # output = subprocess.check_output(cmd, env=env, stderr=subprocess.STDOUT)
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error download model repo: {repo_id}")
            logging.error(f"Error: {str(e)}")
            return folder_location, e
    
    return folder_location, None


def extract_profile_components(data, target_profile_id):
    """
    Extract repoID and src files from components for a given profileID.
    
    Args:
        yaml_data (str): The YAML data as a string
        target_profile_id (str): The profileID to search for
        
    Returns:
        dict: A dictionary containing:
            - 'found' (bool): Whether the profile was found
            - 'components' (list): List of dictionaries with 'repo_id' and 'files' for each component
    """
    result = {
        'found': False,
        'components': [],
        'ngcMetadata': None,
    }

    # Iterate through models
    for model in data.get('models', []):
        # Iterate through variants
        for variant in model.get('modelVariants', []):
            # Iterate through profiles
            for profile in variant.get('optimizationProfiles', []):
                # Check if this is the target profileID
                if profile.get('profileId') == target_profile_id:
                    result['found'] = True
                    result['ngcMetadata'] = profile.get('ngcMetadata', None)
                    # Look for ngcMetadata which contains the workspace components
                    for sha_key, metadata in profile.get('ngcMetadata', {}).items():
                        if 'workspace' in metadata and 'components' in metadata['workspace']:
                            for component in metadata['workspace']['components']:
                                component_info = {
                                    'destination': component.get('dst', ''),
                                    'repo_id': component.get('src', {}).get('repo_id', '')
                                }
                                
                                # Extract files if they exist
                                files = component.get('src', {}).get('files', [])
                                if files:
                                    # Handle both direct strings and dictionaries with name tags
                                    component_info['files'] = []
                                    for f in files:
                                        if isinstance(f, str):
                                            component_info['files'].append(f)
                                        elif isinstance(f, dict) and f.get('!name'):
                                            component_info['files'].append(f.get('!name'))
                                
                                result['components'].append(component_info)
                    
                    return result  # Return once the profile is found
    
    return result  # Return not found if we get here


def show_help():
    """Display help information and exit."""
    help_text = """
    Description: Fetches information about a model from the Hugging Face Hub and optionally downloads it,
    or uploads model artifacts to cloud storage.

    Examples:
      python script.py gpt2
      python script.py --token YOUR_HF_TOKEN --repo-id facebook/bart-large
      python script.py --download --path ~/my_models --repo-id bert-base-uncased
      python script.py --repo-type dataset --download --repo-id mnist
      python script.py --cloud aws --src /path/to/models/ --dst s3://bucket/path --recursive
      python script.py --cloud pvc --src /path/to/model/hf/meta/llama3.1 --dst s3://bucket/secured-models/hf/meta/llama3.1
      python script.py --cloud azure --src /path/to/model/hf/meta/llama3.1 --account cloudera-customer1 --container data --dst modelregistry/secured-models/hf/meta/llama3.1
    """
    print(help_text)
    sys.exit(1)

def check_requirements(download_model, cloud):
    """Check if the required tools are installed."""
    if download_model:
        try:
            subprocess.run(["huggingface-cli", "version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except (subprocess.SubprocessError, FileNotFoundError):
            print("Error: huggingface-cli is required for downloading but not installed.")
            print("Please install it using pip:")
            print("  pip install huggingface_hub")
            return False
        try:
            subprocess.run(["ngc", "version", "info"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except (subprocess.SubprocessError, FileNotFoundError):
            print("Error: NGC CLI is required for downloading but not installed.")
            print("Please install it ngc cli : https://org.ngc.nvidia.com/setup/installers/cli")
            return False
    if cloud:
        if cloud == "aws" or cloud == "pvc":
            try:
                subprocess.run(["aws", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except (subprocess.SubprocessError, FileNotFoundError):
                print(f"Error: aws-cli is required for uploading to {cloud} but not installed.")
                print("Please install it using pip:")
                print("  pip install awscli")
                return False
        elif cloud == "azure":
            try:
                subprocess.run(["az", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except (subprocess.SubprocessError, FileNotFoundError):
                print("Error: Azure CLI is required for uploading to Azure Blob Storage but not installed.")
                print("Please install it using pip:")
                print("  pip install azure-cli")
                return False
        else:
            print(f"Unsupported cloud provider: {cloud}")
            return False
    
    return True

def validate_repo_type(repo_type):
    """Validate the repository type."""
    valid_types = ["hf", "ngc"]
    if repo_type not in valid_types:
        print(f"Error: Invalid repository type '{repo_type}'")
        print(f"Valid types are: {', '.join(valid_types)}")
        return False
    return True

def download_repo_hf(repo_id, token, download_path):
    """Download a Hugging Face repository using huggingface-cli."""

    print(f"Downloading repository: {repo_id} to {download_path}")
    
    # Create necessary directories
    os.makedirs(os.path.join(download_path, "hf", repo_id, "artifacts"), exist_ok=True)
    
    # Download the repository
    print("Starting download with huggingface-cli...")
    try:
        cmd = ["huggingface-cli", "download", repo_id, "--local-dir", f"{download_path}/hf/{repo_id}/artifacts"]
        if token:
            cmd.extend(["--token", token])
        
        subprocess.run(cmd, check=True)
        print(f"Download completed successfully for {repo_id}")
        return True
    except subprocess.SubprocessError:
        print(f"Error: Failed to download {repo_id}")
        return False


def download_repo(repo_id, token, download_path, repo_type, metadata):
    """Download a repository using huggingface-cli."""
    if repo_type == 'hf':
        return download_repo_hf(repo_id, token, download_path)
    elif repo_type == 'ngc':
        repo_path=extract_last_part(repo_id)
        download_path = os.path.join(download_path, "ngc", repo_path, "artifacts")
        os.makedirs(download_path, exist_ok=True)
        # Implement NGC repository downloading if needed
        for component in metadata['components']:
            print(f"Repo ID: {component['repo_id']}")
            print(f"Destination: {component['destination']}")
            execute_ngc_download_command(component['repo_id'], download_path, component.get('files'))


def get_repo_info_hf(repo_id, token):
    """Get huggingface repository metadata and save it to a file."""
    import requests    
    # Prepare headers for API request
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    # Make the API request
    url = f"https://huggingface.co/api/models/{repo_id}"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse JSON and save to file
        return response.json()
    except (requests.RequestException, json.JSONDecodeError) as e:
        print(f"Failed to fetch huggingface repo metadata for the model: {str(e)}")
        return None

def get_repo_info_ngc(repo_id, spec_file):
    """Get NGC repository metadata and save it to a file."""
    # Implement NGC repository metadata fetching if needed
    spec = load_ngc_spec(spec_file)
    ngcMetadata =  extract_profile_components(spec, repo_id)
    repo_id=repo_id.split(':')
    modelMetadata = get_ngc_model_info(repo_id[0], repo_id[1])
    return ngcMetadata, modelMetadata


def get_repo_info(repo_id, token, repo_type, download_path, ngc_spec):
    """Get repository metadata and save it to a file."""
    if repo_type == 'hf':
        metadata_path = os.path.join(download_path, "hf", repo_id, "metadata")
        metadata = get_repo_info_hf(repo_id, token)
        os.makedirs(metadata_path, exist_ok=True)
        output_file = os.path.join(metadata_path, "metadata.json")
        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    elif repo_type == 'ngc':
        repo_path = extract_last_part(repo_id)
        metadata_path = os.path.join(download_path, "ngc", repo_path, "metadata")
        metadata, modelmetadata =  get_repo_info_ngc(repo_id, ngc_spec)
        metadataToFile = metadata['ngcMetadata']
        os.makedirs(metadata_path, exist_ok=True)
        output_file = os.path.join(metadata_path, "metadata.yaml")
        with open(output_file, 'w') as f:
            yaml.dump(metadataToFile, f, default_flow_style=False, sort_keys=False, allow_unicode=True, Dumper=SafeUnknownDumper)
        if modelmetadata:
            output_file = os.path.join(metadata_path, "modelmetadata.json")
            with open(output_file, 'w') as f:
                json.dump(modelmetadata, f, indent=2)

        # Implement NGC repository metadata fetching if needed

        
    print(f"Saved metadata to {output_file}")
    return metadata


def upload_to_cloud(src, dst, cloud, token=None, recursive=False, endpoint=None, 
                    insecure=False, ca_bundle=None, account=None, container=None,
                    repo_id=None, repo_type=None):
    """Upload files to cloud storage."""
    try:
        if repo_type == "ngc":
            repo_id = extract_last_part(repo_id)
        src = os.path.join(src, repo_type, repo_id)
        dst = dst +"/"+ repo_type +"/"+ repo_id
        if cloud == "aws":
            cmd = ["aws", "s3", "cp", src, f"{dst}/", "--recursive"]
            
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)
            
        elif cloud == "azure":
            cmd = [
                "az", "storage", "blob", "upload-batch",
                "--account-name", account,
                "--destination", container,
                "--destination-path", dst,
                "--source", src
            ]
            if token:
                cmd.extend(["--sas-token", token])
            
            subprocess.run(cmd, check=True)
            
        elif cloud == "pvc":
            cmd = ["aws", "s3"]
            
            if endpoint:
                cmd.extend(["--endpoint", endpoint])
            
            if insecure:
                cmd.append("--no-verify-ssl")
            elif ca_bundle:
                cmd.extend(["--ca-bundle", ca_bundle])
            
            cmd.extend(["cp", src, dst, "--recursive"])
            
            subprocess.run(cmd, check=True)
            
        else:
            print(f"Unsupported cloud provider: {cloud}")
            return False
        
        print(f"Uploaded: {src} -> {dst}")
        return True
    
    except subprocess.SubprocessError as e:
        print(f"Error during upload: {str(e)}")
        return False




def print_models(models: List[Dict[str, str]], title: str = "Models"):
    """Print models in a formatted way."""
    print(f"\n=== {title.upper()} ===")
    if not models:
        print("No models found.")
        return
    
    for i, model in enumerate(models, 1):
        print(f"{i}. {model['name']}")
        print(f"   Display Name: {model['displayName']}")
        print(f"   Category: {model['category']}")
        print(f"   Hub ID: {model['modelHubID']}")
        if model['description']:
            desc = model['description'][:100] + "..." if len(model['description']) > 100 else model['description']
            print(f"   Description: {desc}")
        print()

def print_list(items: List[str], title: str):
    """Print a list of items in a formatted way."""
    print(f"\n=== {title.upper()} ===")
    if not items:
        print(f"No {title.lower()} found.")
        return
    
    for i, item in enumerate(items, 1):
        print(f"{i}. {item}")

def print_detailed_model(model_data: Dict, model_name: str):
    """Print detailed model information."""
    print(f"\n=== DETAILED INFO FOR '{model_name}' ===")
    
    if not model_data:
        print(f"Model '{model_name}' not found.")
        return
    
    print(f"Name: {model_data.get('name', 'N/A')}")
    print(f"Display Name: {model_data.get('displayName', 'N/A')}")
    print(f"Model Hub ID: {model_data.get('modelHubID', 'N/A')}")
    print(f"Category: {model_data.get('category', 'N/A')}")
    print(f"Type: {model_data.get('type', 'N/A')}")
    print(f"Description: {model_data.get('description', 'N/A')}")
    print(f"License: {model_data.get('license', 'N/A')}")
    
    if 'labels' in model_data:
        print(f"Labels: {', '.join(model_data['labels'])}")
    
    if 'modelVariants' in model_data:
        print(f"\nModel Variants ({len(model_data['modelVariants'])}):")
        for i, variant in enumerate(model_data['modelVariants'], 1):
            print(f"  {i}. {variant.get('variantId', 'N/A')}")
            if 'optimizationProfiles' in variant:
                print(f"     Optimization Profiles ({len(variant['optimizationProfiles'])}):")
                for j, profile in enumerate(variant['optimizationProfiles'], 1):
                    profile_id = profile.get('profileId', 'N/A')
                    display_name = profile.get('displayName', 'N/A')
                    framework = profile.get('framework', 'N/A')
                    print(f"       {j}. {profile_id}")
                    print(f"          Display Name: {display_name}")
                    print(f"          Framework: {framework}")




def main():
    parser = argparse.ArgumentParser(description="Hugging Face model management script")
    
    # Help option
    
    # Download options
    parser.add_argument("-t", "--token", default="", help="Token for authentication")
    parser.add_argument("-j", "--json", action="store_true", help="Output raw JSON")
    parser.add_argument("-do", "--download", action="store_true", help="Download the model repository")
    parser.add_argument("-p", "--path", default="./models", help="Path to download model files")
    parser.add_argument("-ri", "--repo-id", help="Repository ID to download")
    parser.add_argument("-rt", "--repo-type", default="hf", help="Repository type (default: hf)")
    
    # Upload options
    parser.add_argument("-c", "--cloud", default="aws", help="Cloud provider (aws, gcp, azure, pvc)")
    parser.add_argument("-s", "--src", help="Source directory for upload")
    parser.add_argument("-d", "--dst", help="Destination path in object storage")
    parser.add_argument("-r", "--recursive", action="store_true", help="Recursively upload folders")
    parser.add_argument("-e", "--endpoint", help="S3 gateway endpoint (Private cloud only)")
    parser.add_argument("-i", "--insecure", action="store_true", help="Allow insecure SSL connections")
    parser.add_argument("-ca", "--ca-bundle", help="Path to custom CA bundle file")
    parser.add_argument("-ac", "--account", help="Account for Azure uploads")
    parser.add_argument("-cn", "--container", help="Container name for Azure uploads")
    parser.add_argument("-ns", "--ngc-spec", help="NGC spec file for downloading")
    
    parser.add_argument('-m', '--model-name', help='Name of the model to query')
    parser.add_argument('-vid', '--variant-id', help='Variant ID (used with model name for specific queries)')
    parser.add_argument('--list-all', action='store_true',help='List all available models')
    parser.add_argument('--list-variants', action='store_true',help='List all variant IDs for the specified model')
    parser.add_argument('--list-profiles', action='store_true',help='List all optimization profile IDs for the specified model (and variant if provided)')
    

    args = parser.parse_args()
    
    # # Show help if requested or no arguments provided
    # if args.help or len(sys.argv) == 1:
    #     show_help()
        # Handle different command combinations
    if args.list_all:
        ngc_spec_file = args.ngc_spec
        parser = ModelParser(ngc_spec_file)
        models = parser.get_all_models()
        print_models(models, "All Models")
        return
    
    elif args.model_name:
        ngc_spec_file = args.ngc_spec
        parser = ModelParser(ngc_spec_file)

        
        if args.list_variants:
            variants = parser.get_model_variant_ids(args.model_name)
            print_list(variants, f"Variants for '{args.model_name}'")
        
        elif args.list_profiles:
            profiles = parser.get_optimization_profile_ids(args.model_name, args.variant_id)
            if args.variant_id:
                title = f"Optimization Profiles for '{args.model_name}' variant '{args.variant_id}'"
            else:
                title = f"Optimization Profiles for '{args.model_name}'"
            print_list(profiles, title)
        
        else:
            # Default: show basic info about the model
            model_data = parser.get_detailed_model_info(args.model_name)
            if model_data:
                model_info = {
                    'name': model_data.get('name', ''),
                    'displayName': model_data.get('displayName', ''),
                    'modelHubID': model_data.get('modelHubID', ''),
                    'category': model_data.get('category', ''),
                    'description': model_data.get('description', '')
                }
                print_models([model_info], f"Model '{args.model_name}'")
            else:
                print(f"Model '{args.model_name}' not found.")
        return


    # Check requirements
    if not check_requirements(args.download, args.cloud if args.src else None):
        sys.exit(1)
    
    # Handle download use case
    if args.download:
        if not args.repo_id:
            print("Error: --repo-id is required for download")
            sys.exit(1)
        
        if not validate_repo_type(args.repo_type):
            sys.exit(1)
        
        # Get repository info and download
        metadata = get_repo_info(args.repo_id, args.token, args.repo_type, args.path, args.ngc_spec)
        if metadata is not None:
            download_repo(args.repo_id, args.token, args.path, args.repo_type, metadata)
        else:
            print("Error: Failed to get repository metadata")
            sys.exit(1)
        sys.exit(0)
    
    # Handle upload use case
    if args.src and args.dst:
        if not upload_to_cloud(
            args.src, args.dst, args.cloud, args.token, args.recursive,
            args.endpoint, args.insecure, args.ca_bundle, args.account, args.container,
            args.repo_id, args.repo_type

        ):
            sys.exit(1)
        print("Upload completed.")
    else:
        print("Error: Missing required parameters.")
        show_help()

if __name__ == "__main__":
    main()