import copy
import os
from xml.dom import minidom as dom


class XMLGenerator(object):
    def __init__(self, xml_name: str):
        self.doc = dom.Document()
        self.xml_name = xml_name

    def create_append_node(self, node_name, root_node=None):
        """创建一个新node并将node添加到root_node下"""
        new_node = self.doc.createElement(node_name)
        if root_node is not None:
            root_node.appendChild(new_node)
        else:
            self.doc.appendChild(new_node)
        return new_node

    def create_text_node(self, node_name, node_value, root_node):
        """
        创建一个新node，然后在该node中添加一个text_node，
        最后将node添加到root_node下
        """
        new_node = self.doc.createElement(node_name)
        node_data = self.doc.createTextNode(node_value)
        new_node.appendChild(node_data)
        root_node.appendChild(new_node)

    def create_object_node(self, info_dict: dict = None, root_node: str = None):
        if (info_dict is None) or (root_node is None):
            return

        object_node = self.create_append_node('object', root_node)
        box_node = self.create_append_node('bndbox', object_node)
        self.create_text_node("xmin", info_dict.pop("xmin"), box_node)
        self.create_text_node("ymin", info_dict.pop("ymin"), box_node)
        self.create_text_node("xmax", info_dict.pop("xmax"), box_node)
        self.create_text_node("ymax", info_dict.pop("ymax"), box_node)

        for k, v in info_dict.items():
            self.create_text_node(k, v, object_node)

    def save_xml(self):
        f = open(self.xml_name, "w")
        self.doc.writexml(f, addindent="\t", newl="\n")
        f.close()


def create_pascal_voc_xml(filename: str = None,
                          years: str = 'VOC2012',
                          source_dict: dict = None,
                          objects_list: list = None,
                          im_shape: tuple = None,
                          save_root: str = os.getcwd(),
                          cover: bool = False):
    if not (filename and source_dict and objects_list and im_shape):
        return

    # 0--Parade/0_Parade_marchingband_1_849.jpg -> 0_Parade_marchingband_1_849.xml
    xml_name = filename.split(os.sep)[-1].split(".")[0] + '.xml'
    xml_full_path = os.path.join(save_root, xml_name)
    if os.path.exists(xml_full_path) and (cover is False):
        print(f"{xml_full_path} already exist, skip.")
        return

    xml_generator = XMLGenerator(xml_full_path)

    # xml root node
    node_root = xml_generator.create_append_node('annotation')
    xml_generator.create_text_node(node_name='folder', node_value=years, root_node=node_root)
    xml_generator.create_text_node(node_name='filename', node_value=filename, root_node=node_root)

    # source
    node_source = xml_generator.create_append_node('source', root_node=node_root)
    xml_generator.create_text_node(node_name='database', node_value=source_dict['database'], root_node=node_source)
    xml_generator.create_text_node(node_name='annotation', node_value=source_dict['annotation'], root_node=node_source)
    xml_generator.create_text_node(node_name='image', node_value=source_dict['image'], root_node=node_source)

    # size
    node_size = xml_generator.create_append_node('size', root_node=node_root)
    xml_generator.create_text_node(node_name='height', node_value=str(im_shape[0]), root_node=node_size)
    xml_generator.create_text_node(node_name='width', node_value=str(im_shape[1]), root_node=node_size)
    xml_generator.create_text_node(node_name='depth', node_value=str(im_shape[2]), root_node=node_size)

    # segmented
    xml_generator.create_text_node(node_name='segmented', node_value='0', root_node=node_root)

    # object
    for i, ob in enumerate(objects_list):
        xml_generator.create_object_node(info_dict=ob, root_node=node_root)

    # XML write
    xml_generator.save_xml()


def create_xml_test():
    objects = []
    ob = {'name': 'person', 'pose': 'Unspecified', 'truncated': '0', 'difficult': '0',
          'xmin': '174', 'ymin': '101', 'xmax': '349', 'ymax': '351'}
    objects.append(ob)
    objects.append(copy.deepcopy(ob))

    years = 'VOC2012'
    filename = 'test.jpg'
    source_dict = {'database': 'The VOC2007 Database', 'annotation': 'PASCAL VOC2007', 'image': 'flickr'}
    im_width = '500'
    im_height = '700'
    im_depth = '3'
    im_shape = (im_width, im_height, im_depth)
    create_pascal_voc_xml(filename=filename, years=years,
                          source_dict=source_dict, objects_list=objects,
                          im_shape=im_shape)
