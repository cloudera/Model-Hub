import yaml
# Define a custom YAML loader that preserves unknown tags



import sys
import yaml
import pprint


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

def load_ngc_spec():
    with open('sample.yaml', "r") as file:
        yaml_data = file.read()
    # class PreservingLoader(yaml.SafeLoader):
    #     def ignore_unknown(self, node):
    #       return node.tag, node.value
    
    # PreservingLoader.add_constructor(None, PreservingLoader.ignore_unknown)

    # data = yaml.load(yaml_data,Loader=PreservingLoader)
    data = yaml.load(yaml_data, Loader=MySafeLoader)
    return data


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
    modelCard = ''
    # Iterate through models
    for model in data.get('models', []):
        # Iterate through variants
        for variant in model.get('modelVariants', []):
            modelCard = variant.get('modelCard', '')
            # Iterate through profiles
            for profile in variant.get('optimizationProfiles', []):
                # Check if this is the target profileID
                if profile.get('profileId') == target_profile_id:
                    print("result foung", profile.get('profileId'))
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
                    print("result in side", result)
                    return result, modelCard  # Return once the profile is found
    print("resut is ", result)
    return result , modelCard


def canusenimcli(ngcmetadata):
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
        hash_key = next(iter(ngcmetadata))
        release_info = ngcmetadata[hash_key]
        print(release_info)
        # Check if the release key is present
        if "release" not in release_info:
            return False
        
        if "release" in release_info:
            version = release_info["release"]
            checkversion = "1.3.0"
            from packaging.version import Version
            return Version(version) > Version(checkversion)
    except Exception as e:
        return {"error": f"Failed to parse YAML: {str(e)}"}

# Usage example
if __name__ == "__main__":
    yaml_file_path = "your_file.yaml"  # Replace with your actual file path
    try:
        yaml_data = load_ngc_spec()
        result, modelCard=extract_profile_components(yaml_data,'nim/nvidia/llama-3.3-nemotron-super-49b-v1.5:l40sx4-throughput-bf16-lb51ks7uxa')
        print("modelCard in main", modelCard)
        metadataToFile=result['ngcMetadata']
        print('metadataToFile', metadataToFile)
        print("can use nimcli", canusenimcli(metadataToFile))
        with open('output.yaml', 'w') as f:
          yaml.dump(metadataToFile, f,allow_unicode=True, Dumper=SafeUnknownDumper, default_flow_style=False)
    except FileNotFoundError:
        print(f"Error: File '{yaml_file_path}' not found")
    except Exception as e:
        print(f"Error loading YAML: {e}")