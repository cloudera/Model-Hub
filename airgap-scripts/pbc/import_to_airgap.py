#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
import logging

import yaml

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


def execute_nim_download_command(repo_id, folder_location, ngc_spec, profile_sha, version):
    """
    Execute nim cli download command to download model files.
    
    Args:
        repo_id (str): Repository ID
    """
    model_name = repo_id.split(":")[0]
    count = model_name.count('/')
    if count != 2:
        raise ValueError(f"Expected 3 '/' characters, but found {count} in model name") 
    manifest_path = f"{ngc_spec}/{version}/{model_name}.yaml"
    cmd = [
        "nimcli", "download", "--profiles", profile_sha, "--manifest-file",
        manifest_path, "--model-cache-path", folder_location
    ]

    logging.info(cmd)
        
    try:
        # output = subprocess.check_output(cmd, env=env, stderr=subprocess.STDOUT)
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error download model repo: {repo_id}")
        logging.error(f"Error: {str(e)}")
        logging.error(f"Command output: {e.output.decode()}")
        return folder_location, Exception(e.output.decode())


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
                output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                logging.error(f"Error download model repo: {repo_id}")
                logging.error(f"Error: {str(e)}")
                logging.error(f"Command output: {e.output.decode()}")
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
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error download model repo: {repo_id}")
            logging.error(f"Error: {str(e)}")
            logging.error(f"Command output: {e.output.decode()}")
            return folder_location, Exception(e.output.decode())
    
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


def canusenimcli(metadata):
    """
    Parse YAML string and extract release information.
    
    Args:
        yaml_str (str): YAML string containing release data
        
    Returns:
        dict: Dictionary containing parsed release information
    """
    try:
        # Parse the YAML string

        
        # Extract information from the first entry in the YAML
        # (In this case, there's only one entry with a hash as the key)
        print("here")
        ngcmetadata = metadata['ngcMetadata']
        
        print(ngcmetadata)
        hash_key = next(iter(ngcmetadata))
        print("hashkey")
        release_info = ngcmetadata[hash_key]
        print(release_info)
        # Check if the release key is present
        if "release" not in release_info:
            return False
        
        if "release" in release_info:
            version = release_info["release"]
            checkversion = "1.3.0"
            from packaging.version import Version
            return Version(version) >= Version(checkversion)
    except Exception as e:
        print("error Failed to parse YAML:", e)
        return False

def download_repo(repo_id, token, download_path, repo_type, metadata, ngc_spec, version):
    """Download a repository using huggingface-cli."""
    print(f"start downloading {repo_type} repository: {repo_id}")
    if repo_type == 'hf':
        return download_repo_hf(repo_id, token, download_path)
    elif repo_type == 'ngc':
        ngcPrefix = 'ngc://'
        if repo_id.startswith(ngcPrefix):
            repo_id = repo_id[len(ngcPrefix):]
        repo_path=extract_last_part(repo_id)
        download_path = os.path.join(download_path, "ngc", repo_path, "artifacts")
        os.makedirs(download_path, exist_ok=True)
        # Implement NGC repository downloading if needed
        nimcli=canusenimcli(metadata)
        ngcmetadata = metadata['ngcMetadata']
        profile_sha = next(iter(ngcmetadata))
        if nimcli:
            model_name = repo_id.split(":")[0]
            model_name_parts = model_name.split("/")
            if len(model_name_parts)!=3:
                raise NameError("Model name should have three parts "+model_name)
            execute_nim_download_command(repo_id, download_path, ngc_spec, profile_sha, version)
        else:
            for component in metadata['components']:
                print(f"Repo ID: {component['repo_id']}")
                print(f"Destination: {component['destination']}")
                execute_ngc_download_command(component['repo_id'], download_path, component.get('files'))
    print("Finish downloading artifacts")


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


def get_repo_info(repo_id, token, repo_type, download_path, ngc_spec, version):
    """Get repository metadata and save it to a file."""
    print(f"Fetching information for {repo_type} repository: {repo_id}")
    print(f"Download path: {download_path}")
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
        ngc_spec_file = f'{ngc_spec}/{version}/ngc-spec.yaml'
        metadata, modelmetadata =  get_repo_info_ngc(repo_id, ngc_spec_file)
        metadataToFile = metadata['ngcMetadata']
        os.makedirs(metadata_path, exist_ok=True)
        output_file = os.path.join(metadata_path, "metadata.yaml")
        with open(output_file, 'w') as f:
            yaml.dump(metadataToFile, f, default_flow_style=False, sort_keys=False, allow_unicode=True, Dumper=SafeUnknownDumper)
        if modelmetadata:
            output_file = os.path.join(metadata_path, "modelmetadata.json")
            with open(output_file, 'w') as f:
                json.dump(modelmetadata, f, indent=2)
    print("finish downloading metadata file")

        # Implement NGC repository metadata fetching if needed

        
    print(f"Saved metadata to {output_file}")
    return metadata


def upload_to_cloud(src, dst, cloud, token=None, recursive=False, endpoint=None, 
                    insecure=False, ca_bundle=None, account=None, container=None,
                    repo_id=None, repo_type=None):
    """Upload files to cloud storage."""

    print(f"Start uploading  {repo_type} repository: {repo_id}")
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
        print(f"Finish uploading  {repo_type} repository: {repo_id} to {cloud}")
        return True
    
    except subprocess.SubprocessError as e:
        print(f"Error during upload: {str(e)}")
        return False

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
    # parser.add_argument("-sha", "--profile-sha", help="Sha of the p rofile of the repoID")
    
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
    parser.add_argument("-ns", "--ngc-spec", help="NGC spec folder path")
    parser.add_argument("-v", "--version", help="Version of AI registry")

    
    args = parser.parse_args()
    
    # # Show help if requested or no arguments provided
    # if args.help or len(sys.argv) == 1:
    #     show_help()
    
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
        metadata = get_repo_info(args.repo_id, args.token, args.repo_type, args.path, args.ngc_spec, args.version)
        if metadata is not None:
            download_repo(args.repo_id, args.token, args.path, args.repo_type, metadata, args.ngc_spec, args.version)
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