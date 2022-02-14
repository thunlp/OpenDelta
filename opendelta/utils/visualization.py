from typing import List
from rich.tree import Tree as RichTree
from rich import print as richprint
import torch
import torch.nn as nn
import re
from collections import OrderedDict
import opendelta.utils.logging as logging
logger = logging.get_logger(__name__)
class ModuleTree(RichTree):
    def __init__(
        self,
        module_name=None,
        info=None,
        is_param_node=False,
        type_color="green",
        param_color="red",
        main_color="white",
        style = "tree",
        guide_style = "tree.line",
        expanded=True,
        highlight=False,
        ):
        self.module_name = module_name
        self.info = info
        self.is_param_node = is_param_node
        self.type_color = type_color
        self.param_color = param_color
        self.main_color = main_color
        label = self.set_label()
        super().__init__(label,style=style,guide_style=guide_style,expanded=expanded,highlight=highlight)


    def add(
        self,
        module_name=None,
        info=None,
        is_param_node=False,
        type_color="green",
        param_color="red",
        main_color="white",
        style=None,
        guide_style=None,
        expanded=True,
        highlight=False,
        ):
        node = ModuleTree(
            module_name,
            info,
            is_param_node,
            type_color,
            param_color,
            main_color,
            style=self.style if style is None else style,
            guide_style=self.guide_style if guide_style is None else guide_style,
            expanded=expanded,
            highlight=self.highlight if highlight is None else highlight,
        )
        self.children.append(node)
        return node
    
    def set_label(self):
        if self.module_name is not None:
            label = f"[{self.main_color}]{self.module_name}"
        else:
            label = ""
        if self.info is not None:
            if not self.is_param_node:
                label += f" [{self.type_color}]({self.info})"
            else:
                label += f" [{self.param_color}]{self.info}"
        self.label = label
        return label


class Visualization(object):
    r"""
    Better visualization tool for *BIG* pretrained models.
    
    - Better repeated block representation
    - Clearer parameter position 
    - and Visible parameter state. 

    Args:
        plm (:obj:`torch.nn.Module`): The pretrained model, actually can be any pytorch module.

    """
    def __init__(self, plm: nn.Module):

        self.plm = plm
        self.type_color = "green"
        self.param_color = "cyan"
        self.duplicate_color = "red"
        self.normal_color = "white"
        self.virtual_color = "orange"
        self.not_common_color = "bright_black"
        self.no_grad_color = "rgb(0,70,100)"
        self.delta_color = "rgb(175,0,255)"

    def check_mode(self, ):
        if self.keep_non_params and self.common_structure:
            raise RuntimeError("keep_non_params can't be used will common_structure. The common structure only contains parameter nodes.")
        if self.common_structure:
            if self.mapping is None:
                raise RuntimeError("Mapping hasn't been given.")

    def structure_graph(self, 
                        rootname="root", 
                        expand_params=False, 
                        keep_non_params=False, 
                        common_structure=False, 
                        mapping=None, 
                        only_common=False,
                        printTree=True,
                        ):
        r"""Draw the structure graph in command line. 

        Args: 
            rootname (:obj:`str`) The root node's name. 
            keep_non_params (:obj:`bool`) Display the modules that does not have parameters, such as nn.Dropout 
            expand_params (:obj:`bool`) Display parameter infomation (shape, etc) in seperate lines. "
            common_structure (:obj:`bool`) Whether convert the structure into a common structure defined in structure_mapping.py. The not common structure will be displayed in grey.
            only_common (:obj:`bool`) Whether ignore the modules that are not in common structure. This will result in a more compact view. Default to False.
            mapping (:obj:`dict`) The structure mapping. Must provide if common_structure=True.
        """

        self.keep_non_params = keep_non_params
        self.expand_params = expand_params
        self.rootname = rootname
        self.only_common = only_common
        self.common_structure = common_structure
        self.mapping = mapping
        self.check_mode()
        # root_tree = self.build_tree(rootname)
        self.root_tree = ModuleTree(self.rootname)
        if common_structure:
            self.build_common_tree(self.plm, mapping, self.root_tree)
        else:
            self.build_tree(self.plm, self.root_tree)    
        self.prune_tree(self.root_tree)
        if not self.expand_params:
            self.fold_param_node(self.root_tree)
        if printTree:
            richprint(self.root_tree)
        return self.root_tree

    

        
    def is_leaf_module(self, module):
        r"""[NODOC] Whether the module is a leaf module
        """
        return len([n for n,_ in module.named_children()]) == 0
        
    def build_tree(self, module:nn.Module, tree:ModuleTree=None):
        r"""[NODOC] build the originial tree structure
        """
        if self.is_leaf_module(module):
            return 
        else:
            for n,m in module.named_children():
                type_info = re.search(r'(?<=\').*(?=\')', str(type(m))).group()
                type_info = type_info.split(".")[-1]
                newnode = tree.add(n, info=type_info, type_color=self.type_color)
                self.add_param_info_node(m, newnode)
                self.build_tree(module=m, tree=newnode)

    def has_parameter(self, module):
        return len([p for p in module.parameters()])>0


    def build_common_tree(self, module:nn.Module, mapping, tree:ModuleTree=None, query="", key_to_root=""):
        r""" (Unstable) build the common tree structure
        """
        if self.is_leaf_module(module): 
            if len(query)>0: # the field is not in mapping
                if self.has_parameter(module):
                    # from IPython import embed
                    # embed(header = "in leaf")
                    logger.warning(f"Parameter node {query} not found under tree {tree.module_name} and module {module}. Is your mapping correct?")  # WARNING
            return 
        else:
            for n,m in module.named_children():
                new_query = query+n
                type_info = re.search(r'(?<=\').*(?=\')', str(type(m))).group()
                type_info = type_info.split(".")[-1]
                if new_query in mapping or "$" in mapping:
                    # print("query",new_query)
                    # from IPython import embed
                    # embed()
                    if new_query in mapping:
                        new_mapping = mapping[new_query]
                        name = new_mapping["__name__"]
                        if len(name.split(".")) > 1: # new key contains a hierarchy , then unfold the hierarchy.
                            # insert virtual node
                            hierachical_name = name.split(".")
                            temp_tree = self.find_or_insert(tree, hierachical_name)
                            newnode = temp_tree.add(hierachical_name[-1], info=type_info, type_color=self.type_color)
                        elif name=="": # the key not in a predefined common structure
                            if self.only_common:
                                continue
                            else: # add the originial name into the tree 
                                newnode = tree.add(new_query, info=type_info, main_color=self.not_common_color, type_color=self.not_common_color)
                        else: # a single new key
                            newnode = self.find_not_insert(tree, [name,""]) # try to find the node
                            if newnode is not None:
                                newnode.info = type_info
                                newnode.type_color = self.type_color
                                newnode.set_label()
                            else:
                                newnode = tree.add(name, info=type_info, type_color=self.type_color)
                    elif "$" in mapping: # match any thing in the field.
                        new_mapping = mapping["$"]
                        newnode = tree.add(n, info=type_info, type_color=self.type_color)
                    self.add_param_info_node(m, newnode)
                    self.build_common_tree(module=m, tree=newnode, mapping=new_mapping, key_to_root=key_to_root+"."+new_query)
                else: 
                    # try to find from root
                    # trsf_key = transform(key_to_root.strip("."), self.mapping)
                    # parent_node = self.find_not_insert(self.root_tree, trsf_key.split(".")+[""])
                    # if parent_node is not None:
                    #     new_mapping = mapping[new_query]
                    #     newnode = parent_node.add(name, info=type_info, type_color=self.type_color)
                    #     self.build_common_tree(module=m, tree=parent_node, mapping )
                    # print("notin query",new_query)
                    # if new_query == "dense":
                    #     from IPython import embed
                    #     embed()
                    # print(f"::{query},,{new_query}, {list(mapping.keys())}")
                    new_query += "."
                    self.build_common_tree(module=m, tree=tree, mapping=mapping, query=new_query, key_to_root=key_to_root)


                
    def find_or_insert(self, tree:ModuleTree, hierachical_name:List[str] ):
        r"""[NODOC] Find the node, if not find, insert a virtual node
        """
        if len(hierachical_name)==1:
            return tree
        names = [x.module_name for x in tree.children]
        if hierachical_name[0] not in names:
            new_node = tree.add(hierachical_name[0], info="Virtual", type_color=self.virtual_color)
        else:
            for x in tree.children:
                if x.module_name == hierachical_name[0]:
                    new_node = x
                    break
        return self.find_or_insert(new_node, hierachical_name=hierachical_name[1:])
    
    def find_not_insert(self, tree:ModuleTree, hierachical_name:List[str] ):
        r"""[NODOC] Find the node but not insert
        """
        if len(hierachical_name)==1:
            return tree
        names = [x.module_name for x in tree.children]
        if hierachical_name[0] not in names:
            return None
        else:
            for x in tree.children:
                if x.module_name == hierachical_name[0]:
                    new_node = x
                    break
        return self.find_not_insert(new_node, hierachical_name=hierachical_name[1:])


        
    def fold_param_node(self, t: ModuleTree, p:ModuleTree=None):
        r"""[NODOC] place the parameters' infomation node right after the module that contains the parameters.
        E.g. w1 (Linear)
             -- weight: [32128, 1024]
        =>
             w1 (Linear) weight: [32128, 1024]
        
        """
        if hasattr(t,"is_param_node") and t.is_param_node:
            p.label += t.label
            return True # indicate whether should be removed
        elif len(t.children) == 0:
            if self.keep_non_params:
                return False
            else:
                return True
        else:
            rm_idx = []
            for idx, c in enumerate(t.children):
                if self.fold_param_node(t=c, p=t):
                    rm_idx.append(idx)
            t.children = [t.children[i] for i in range(len(t.children)) if i not in rm_idx]
            return False

    def prune_tree(self, t: ModuleTree):
        r"""[NODOC] Calculate the _finger_print of a module as the _finger_print of all child node plus the _finger_print of itself.
        The leaf node will have the _finger_print == label. 
        Merge the different node that as the same _finger_print into a single node. 
        """
        if len(t.children) == 0:
            setattr(t, "_finger_print", t.label)
            return

        for idx, sub_tree in enumerate(t.children):
            self.prune_tree(sub_tree)

        t_finger_print = t.label +"::"+";".join([x._finger_print for x in t.children])
        setattr(t, "_finger_print", t_finger_print)
        
        nohead_finger_print_dict = OrderedDict()
        for child_id, sub_tree in enumerate(t.children):
            fname_list = sub_tree._finger_print.split("::")
            if len(fname_list)==1:
                fname = fname_list[0]
            else:
                fname = "::".join(fname_list[1:])
            if fname not in nohead_finger_print_dict:
                nohead_finger_print_dict[fname] = [child_id]
            else:
                nohead_finger_print_dict[fname].append(child_id)

        new_childrens = []
        for groupname in nohead_finger_print_dict:
            representative_id = nohead_finger_print_dict[groupname][0]
            representative = t.children[representative_id]
            group_node = [t.children[idx] for idx in nohead_finger_print_dict[groupname]]
            
            representative = self.extract_common_and_join(group_node)
            new_childrens.append(representative)
        t.children = new_childrens


    def extract_common_and_join(self, l:List[ModuleTree]):
        r"""[NODOC] Some modules that have the same info (e.g., are all "Linear") have different names (e.g., w1,w2)
        Merge them.
        E.g. tree1.module_name = "w1", tree1.info = "Linear"; tree2.module_name = "w1", tree2.info = "Linear" 
        -> representive.module_name = "w1,w2", representive.info = "Linear"
        """
        representative = l[0]
        if len(l)==1:
            return representative
        name_list = [x.module_name for x in l]
        info_list = [x.info for x in l]
        type_hint_dict = OrderedDict()
        for x, y in zip(name_list, info_list):
            if y not in type_hint_dict:
                type_hint_dict[y] = [x]
            else:
                type_hint_dict[y].append(x)
        
        s = ""
        names = ""
        typeinfos = ""
        for t in type_hint_dict:
            group_components = type_hint_dict[t]
            group_components = self.neat_expr(group_components)
            names += group_components+","
            typeinfos += t+","
            s += f"[{self.duplicate_color}]{group_components}[{self.type_color}]({t})"
            s += f","
        names = names[:-1]
        s = s[:-1]
        typeinfos = typeinfos[:-1]
        representative.module_name = names
        representative.type_info = typeinfos
        representative.label = s
        return representative

    def neat_expr(self, l:List[str]):
        r"""[NODOC] A small tool function to arrange the consecutive number into interval display.
        E.g., ["1","2","3","5","6","9","10","11","12"] -> ["1-3","5-6","9-12"]
        """
        try:
            s = self.ranges([int(x.strip()) for x in l])
            s = [str(x)+"-"+str(y) for x,y in s]
            return ",".join(s)
        except:
            return ",".join(l)
    
    def ranges(self, nums:List[int]):
        r"""[NODOC] A small tool function to arrange the consecutive number into interval display.
        E.g., [1,2,3,5,6,9,10,11,12] -> [[1,3],[5,6],[9,12]]
        """
        nums = sorted(set(nums))
        gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
        edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
        return list(zip(edges, edges))

    def add_param_info_node(self, m:nn.Module, tree:ModuleTree, record_grad_state=True, record_delta=True):
        r"""[NODOC] Add parameter infomation of the module. The parameters that are not inside a module (i.e., created using nn.Parameter) will be added in this function.
        """
        known_module = [n for n,c in m.named_children()]
        for n,p in m.named_parameters():
            if n.split(".")[0] not in known_module:
                if len(n.split(".")) > 1: raise RuntimeError(f"The name field {n} should be a parameter since it doesn't appear in named_children, but it contains '.'")
                info = "{}:{}".format(n, list(p.shape))

                if record_grad_state:
                    if not p.requires_grad:
                        color = self.no_grad_color
                    else:
                        color = self.param_color
                else:
                    color = self.param_color
                
                if record_delta:
                    if hasattr(p, "_is_delta") and getattr(p, "_is_delta"):
                        color = self.delta_color

                tree.add(info=info, is_param_node=True, param_color=color)

    
 
        
    


if __name__=="__main__":
    # example command line: 
    # 1. python opendelta/utils/visualization.py --model t5-lm --model_name_or_path t5-large-lm-adapt --common_structure --only_common
    # 2. python opendelta/utils/visualization.py --model roberta --model_name_or_path roberta-large --common_structure 
    # 3. python opendelta/utils/visualization.py --model gpt2 --model_name_or_path gpt2-medium --keep_non_params --expand_params 
    from openprompt.plms import load_plm
    import argparse
    parser = argparse.ArgumentParser("")
    parser.add_argument("--model", type=str, default='t5-lm', help="We test both t5 and t5-lm in this scripts, the corresponding tokenizerwrapper will be automatically loaded.")
    parser.add_argument("--model_name_or_path", default="t5-large-lm-adapt")
    parser.add_argument("--cache_base", default='/home/hushengding/plm_cache/')
    parser.add_argument("--keep_non_params", action="store_true", help="Display the modules that does not have parameters, such as nn.Dropout")
    parser.add_argument("--expand_params", action="store_true", help="Display parameter infomation (shape, etc) in seperate lines. ")
    parser.add_argument("--common_structure", action="store_true", help="Whether convert the structure into a common structure defined in structure_mapping.py. The not common structure will be displayed in grey." )
    parser.add_argument("--only_common", action="store_true", help="Whether ignore the modules that are not in common structure. This will result in a more compact view. Default to False")
    args = parser.parse_args()
    plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.cache_base+args.model_name_or_path)
    print("Model Loaded!")
    if args.common_structure:
        from opendelta.utils.structure_mapping import Mappings
        mapping = Mappings[args.model]
    else:
        mapping = None
    visobj = Visualization(plm)
    visobj.structure_graph(rootname=args.model_name_or_path, keep_non_params=args.keep_non_params, expand_params=args.expand_params, common_structure=args.common_structure, only_common=args.only_common, mapping=mapping)
