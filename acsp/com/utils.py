#coding=utf-8


import os
import math
import datetime


def get_file_ids(path):
    """Get all file names in path."""

    # print "get_fileids path:%s" % path
    file_ids = []
    for root, dirs, files in os.walk(path):
        file_ids.extend(files)
    file_ids = [f for f in file_ids if not f.startswith('.')]

    return file_ids


def get_dirs(path):
    """Get all directories in path."""

    list_dirs = []
    for root, dirs, files in os.walk(path):
        list_dirs.extend(dirs)
    return list_dirs


def make_easy_tag(dom, tagname, value=None, tagtype='text'):
    """Make tag for xml"""

    tag = dom.createElement(tagname)
    if value is not None:

        if value.find(']]>') > -1:
            tagtype = 'text'

        if tagtype == 'text':
            value = value.replace('&', '&amp;')
            value = value.replace('<', '&lt;')
            text = dom.createTextNode(value)
        elif tagtype == 'cdata':
            text = dom.createCDATASection(value)
        tag.appendChild(text)
    return tag


def indent(dom, node, idnt=0):
    """Intent xml"""

    # Copy child list because it will change soon
    children = node.childNodes[:]
    # Main node doesn't need to be indented
    if idnt:
        text = dom.createTextNode('\n' + '\t' * idnt)
        node.parentNode.insertBefore(text, node)
    if children:
        # Append newline after last child, except for text nodes
        if children[-1].nodeType == node.ELEMENT_NODE:
            text = dom.createTextNode('\n' + '\t' * idnt)
            node.appendChild(text)
        # Indent children which are elements
        for n in children:
            if n.nodeType == node.ELEMENT_NODE:
                indent(dom, n, idnt + 1)


def get_tag_text(root, tagname):
    """Get text by tagname."""

    node = root.getElementsByTagName(tagname)[0]
    rc = ''
    for node in node.childNodes:
        if node.TEXT_NODE == node.nodeType:
            rc = rc + node.data
    return rc


def get_tag_list(root, tagname):
    """Get all the sub tags by tagname."""

    node = root.getElementsByTagName(tagname)[0]
    lst = []
    for n in node.childNodes:
        if n.ELEMENT_NODE == n.nodeType:
            if not n.childNodes:
                lst.append('')
            else:
                if n.TEXT_NODE == n.childNodes[0].nodeType:
                    lst.append(n.childNodes[0].data)

    return lst


def num_per_chunk_by_element(l, n):
    """
    chunks_by_element(range(10),3)
    [3, 3, 4]
    """
    l = int(l)
    n = int(n)

    if l > n:
        mode = l % n
        quotient = l / n
        return [n]*int(quotient-1) + [mode+n]
    else:
        return [l]


def chunks_by_element(arr, n):
    """
    chunks_by_element(range(10),3)
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    """
    return [arr[i:i+n] for i in range(0, len(arr), n)]


def num_per_chunk_by_piece(l, m):
    """
    num_per_chunk_by_piece(7,2)
    [3, 4]
    """
    l = int(l)
    m = int(m)

    if l > m:
        quotient = l / m
        return [quotient]*(m-1) + [l-quotient*(m-1)]
    else:
        return [l]


def chunks_by_piece(arr, m):
    """
    chunks_by_piece(range(10),3)
    [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    """
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]


def print_time(time_str):
    """Print time"""
    print("%s:%s" % (time_str, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
